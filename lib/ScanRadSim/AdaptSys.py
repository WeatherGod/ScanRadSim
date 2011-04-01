from task import StaticJob
from NDIter import ChunkIter
import numpy as np
from datetime import timedelta, datetime

_sensing_sys = {}
def register_sensing(sysClass) :
    if sysClass.name not in _sensing_sys :
        _sensing_sys[sysClass.name] = sysClass
    else :
        raise ValueError("The %s is already registered by %s" % (sysClass.name, _sensing_sys[sysClass.name].__class__))

def adapt(name, volume=None) :
    return _sensing_sys[name](volume)



class AdaptSenseSys(object) :
    """
    Base class for any Adaptive sensing system.
    """
    def __init__(self, volume=None) :
        if volume is None :
            volume = (slice(None), slice(None), slice(None))

        self.volume = volume

    def __call__(self, radData) :
        raise NotImplementedError("This function has to be implemented by the derived AdaptSenseSys class!")

class NullSensingSys(AdaptSenseSys) :
    """
    Does not produce any adaptive scaning jobs.
    """
    name = "Null"
    def __call__(self, radData) :
        return [], []
register_sensing(NullSensingSys)


from scipy.ndimage.measurements import find_objects, label
class SimpleSensingSys(AdaptSenseSys) :
    """
    Just scan for every contiguous +35dBz region.
    """
    name = "Simple"
    def __init__(self, volume=None) :
        self.prevJobs = []
        AdaptSenseSys.__init__(self, volume)

    def __call__(self, radData) :
        # Find the maximum value along each radial.
        maxView = np.nanmax(radData[self.volume], axis=-1)
        features, labels = self._find_features(maxView)
        return self._process_features(radData[self.volume], features)

    def _radial_counts(self, objects) :
        # Assumes last dimension is range-gate
        return [self._radial_cnt(radials[:-1]) for radials in objects]

    def _radial_cnt(self, radials) :
        return int(np.prod([aSlice.stop - aSlice.start for
                            aSlice in radials]))

    def _slice_widths(self, objects) :
        # Assumes second dimension is azimuth
        return [self._radial_cnt(radials[1:2]) for radials in objects]

    def _slice_heights(self, objects) :
        # Assumes first dimension is elevation
        return [self._radial_cnt(radials[0:1]) for radials in objects]

    def _find_features(self, radData) :
        # Assumes first two dims are elevation and azimuth
        labels, cnt = label(radData >= 35.0)

        if cnt == 0 :
            return [], labels

        objects = find_objects(labels)

        # In the following, we will build the list of features that
        # are large enough to keep.  We will also modify the labels
        # array so that the values in there are merely the index
        # number for the radial in allRadials (minus one, of course).
        allRadials = []
        newIndex = 0
        for index, radials in enumerate(objects) :
            # Assumes that the first two dims are elevation and azimuth
            cnt = self._radial_cnt(radials[:2])

            if cnt < 20 :
                # Too small to care...
                labels[radials][labels[radials] == (index + 1)] = 0
            else :
                newIndex += 1
                allRadials.append(radials)
                # Remember, radials covers more area than labeled.
                # So, we only want to impact the relevant labels.
                # We use labels[radials] to help shrink the search area.
                labels[radials][labels[radials] == (index + 1)] = newIndex

        #print len(allRadials), labels.max()
        return allRadials, labels

    def _reform_slices(self, features) :
        # Assumes that the first two dimensions are elevation and azimuth
        # Make it so that the range-gate dimension is sliced in its entirety.
        return [radials[:2] + (slice(None),) for radials in features]

    def _process_features(self, radData, features) :
        jobsToRemove = self.prevJobs

        # The label and find_objects will slice only the
        # relevant radials, but we still need something to
        # represent the entire length of the radials.
        allRadials = self._reform_slices(features)
        widths = self._slice_widths(features)

        gridshape = radData.shape
        
        jobsToAdd = [StaticJob(timedelta(seconds=20),
                               #(radials,),
                               ChunkIter(gridshape, width, radials),
                               dwellTime=timedelta(microseconds=64000),
                               prt=timedelta(microseconds=800)) for
                     radials, width in zip(allRadials, widths)]

        self.prevJobs = jobsToAdd
        #print "Add:", jobsToAdd
        #print "Remove:", jobsToRemove
        return jobsToAdd, jobsToRemove
register_sensing(SimpleSensingSys)



class VolSensingSys(SimpleSensingSys) :
    """
    Just scan for every contiguous +35dBz region, but for the whole 3D volume.
    """
    name = "SimpleVol"
    def __init__(self, volume=None) :
        SimpleSensingSys.__init__(self, volume)

    def __call__(self, radData) :
        features, labels = self._find_features(radData[self.volume])
        return self._process_features(radData[self.volume], features)
register_sensing(VolSensingSys)


class SimpleTrackingSys(VolSensingSys) :
    """
    Scan for every contiguous +35dBz region in the 3D volume.
    Also, utilize a simple overlap tracking algorithm to provide tracking data.
    """
    name = "SimpleTracking"
    def __init__(self, volume=None) :
        self._jobRegions = []
        VolSensingSys.__init__(self, volume)

    def __call__(self, radData) :
        features, labels = self._find_features(radData[self.volume])
        return self._process_features(radData[self.volume], features, labels)

    def _process_features(self, radData, features, labels) :
        job2Feature = self._track_features(features, labels)

        jobsToKeep = []
        slicesToKeep = []
        jobsToRemove = []

        gridshape = radData.shape
        allRadials = self._reform_slices(features)
        widths = self._slice_widths(features)

        for featIndex, oldJob in zip(job2Feature, self.prevJobs) :
            if featIndex == -1 :
                jobsToRemove.append(oldJob)
            else :
                oldJob.reset(ChunkIter(gridshape, widths[featIndex], allRadials[featIndex]))
                jobsToKeep.append(oldJob)
                slicesToKeep.append(allRadials[featIndex])

        jobsToAdd = []
        slicesToAdd = []
        for index, (radials, feature, width) in enumerate(zip(allRadials, features, widths)) :
            if index not in job2Feature :
                # An object without a pre-existing job!
                jobsToAdd.append(StaticJob(timedelta(seconds=20),
                               #(radials,),
                               ChunkIter(gridshape, width, radials),
                               dwellTime=timedelta(microseconds=64000),
                               prt=timedelta(microseconds=800)))
                slicesToAdd.append(feature)

        self.prevJobs = jobsToKeep + jobsToAdd
        self._jobRegions = slicesToKeep + slicesToAdd
        return jobsToAdd, jobsToRemove

    def _track_features(self, features, labels) :
        job2Feature = []
        howMuchOverlap = []
        cnt = len(features)

        if cnt == 0 :
            return job2Feature

        for oldJob, oldSlice in zip(self.prevJobs, self._jobRegions) :
            # Ignore zeros with "[1:]".
            labelCnts = np.bincount(labels[oldSlice].flatten(), minlength=cnt + 1)[1:]
            # We also know that there is at least one, so we can go ahead with an argmax
            bestOverlap = np.argmax(labelCnts)
            howMuchOverlap.append(labelCnts[bestOverlap])
            if labelCnts[bestOverlap] == 0 :
                # This old job does not overlap sufficiently with any currently
                # identified features.
                job2Feature.append(-1)
            else :
                if bestOverlap in job2Feature :
                    # This label has already been paired with
                    # an existing job.  Need to determine which to keep.
                    # Keep the bigger one...
                    otherIndex = job2Feature.index(bestOverlap)
                    if howMuchOverlap[otherIndex] < labelCnts[bestOverlap] :
                        # This job has better overlap than the other one.
                        # End the other job.
                        job2Feature[otherIndex] = -1
                        job2Feature.append(bestOverlap)
                    else :
                        # The other job had better match, so end this one instead
                        job2Feature.append(-1)
                else :
                    job2Feature.append(bestOverlap)

        return job2Feature
register_sensing(SimpleTrackingSys)


