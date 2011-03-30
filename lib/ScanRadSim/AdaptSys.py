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
        jobsToRemove = self.prevJobs

        # The label and find_objects will slice only the
        # relevant radials, but we still need something to
        # represent the entire length of the radial.
        fullRange = slice(None, None, None)

        # Find the maximum value along each radial.
        maxView = np.nanmax(radData[self.volume], axis=-1)
        #envTotRads = np.prod(radData[self.volume].shape[:-1])
        labels, cnt = label(maxView >= 35.0)

        if cnt == 0 :
            self.prevJobs = []
            return [], jobsToRemove

        objects = find_objects(labels)

        allRadials = [radials + (fullRange,) for radials in objects]
        radCnts = [int(np.prod([aSlice.stop - aSlice.start for
                                aSlice in radials])) for
                   radials in objects]
        widths = [radials[1].stop - radials[1].start for
                  radials in objects]

        #print "Rad Counts:", radCnts
        # Filter out the small storms
        allRadials = [radials for radials, cnt in zip(allRadials, radCnts) if cnt >= 20]
        widths = [width for width, cnt in zip(widths, radCnts) if cnt >= 20]
        radCnts = [cnt for cnt in radCnts if cnt >= 20]

        #totScanRads = sum(radCnts)
        gridshape = radData[self.volume].shape
        
        jobsToAdd = [StaticJob(timedelta(seconds=20),
                               #(radials,),
                               ChunkIter(gridshape, width, radials),
                               dwellTime=timedelta(microseconds=64000),
                               prt=timedelta(microseconds=800)) for
                     radials, cnt, width in zip(allRadials, radCnts, widths)]

        self.prevJobs = jobsToAdd
        return jobsToAdd, jobsToRemove
register_sensing(SimpleSensingSys)



class VolSensingSys(AdaptSenseSys) :
    """
    Just scan for every contiguous +35dBz region, but for the whole 3D volume.
    """
    name = "SimpleVol"
    def __init__(self, volume=None) :
        self.prevJobs = []
        AdaptSenseSys.__init__(self, volume)

    def __call__(self, radData) :
        jobsToRemove = self.prevJobs

        # The label and find_objects will slice only the
        # relevant radials, but we still need something to
        # represent the entire length of the radial.
        fullRange = slice(None, None, None)

        labels, cnt = label(radData[self.volume] >= 35.0)

        if cnt == 0 :
            self.prevJobs = []
            return [], jobsToRemove

        objects = find_objects(labels)

        allRadials = [radials[:-1] + (fullRange,) for radials in objects]
        radCnts = [int(np.prod([aSlice.stop - aSlice.start for
                                aSlice in radials[:-1]])) for
                   radials in objects]
        widths = [radials[1].stop - radials[1].start for
                  radials in objects]

        #print "Rad Counts:", radCnts
        # Filter out the small storms
        allRadials = [radials for radials, cnt in zip(allRadials, radCnts) if cnt >= 20]
        widths = [width for width, cnt in zip(widths, radCnts) if cnt >= 20]
        radCnts = [cnt for cnt in radCnts if cnt >= 20]

        #totScanRads = sum(radCnts)
        gridshape = radData[self.volume].shape
        
        jobsToAdd = [StaticJob(timedelta(seconds=40),
                               #(radials,),
                               ChunkIter(gridshape, width, radials),
                               dwellTime=timedelta(microseconds=64000),
                               prt=timedelta(microseconds=800)) for
                     radials, cnt, width in zip(allRadials, radCnts, widths)]

        self.prevJobs = jobsToAdd
        return jobsToAdd, jobsToRemove
register_sensing(VolSensingSys)

class SimpleTrackingSys(AdaptSenseSys) :
    """
    Just scan for every contiguous +35dBz region in the 3D volume.
    Also, perform a simple "overlap" tracking method to keep jobs alive
    and to kill old jobs.
    """
    name = "SimpleTracking"
    def __init__(self, volume=None) :
        self.prevJobs = []
        self._jobRegions = []
        AdaptSenseSys.__init__(self, volume)

    def __call__(self, radData) :
        # The label and find_objects will slice only the
        # relevant radials, but we still need something to
        # represent the entire length of the radial.
        fullRange = slice(None)

        labels, cnt = label(radData[self.volume] >= 35.0)

        if cnt == 0 :
            jobsToRemove = self.prevJobs
            self.prevJobs = []
            return [], jobsToRemove

        objects = find_objects(labels)

        allRadials = [radials[:-1] + (fullRange,) for radials in objects]
        radCnts = [int(np.prod([aSlice.stop - aSlice.start for
                                aSlice in radials[:-1]])) for
                   radials in objects]
        widths = [radials[1].stop - radials[1].start for
                  radials in objects]

        gridshape = radData[self.volume].shape

        job2Object = []
        jobsToRemove = []
        howMuchOverlap = []
        for oldJob, oldSlice in zip(self.prevJobs, self._jobRegions) :
            # Ignore zeros with "[1:]".
            labelCnts = np.bincount(labels[oldSlice].flat, minlength=cnt + 1)[1:]
            # We also know that there is at least one, so we can go ahead with an argmax
            bestOverlap = np.argmax(labelCnts)
            howMuchOverlap.append(labelCnts[bestOverlap])
            if labelCnts[bestOverlap] == 0 :
                # This old job does not overlap sufficiently with any currently
                # identified features.
                job2Object.append(-1)
                jobsToRemove.append(oldJob)
            else :
                if bestOverlap in job2Object :
                    # This label has already been paired with
                    # an existing job.  Need to determine which to keep.
                    # Keep the bigger one...
                    otherIndex = job2Object.index(bestOverlap)
                    if howMuchOverlap[otherIndex] < labelCnts[bestOverlap] :
                        # This job has better overlap than the other one.
                        # End the other job.
                        oldJob.reset(ChunkIter(gridshape, widths[bestOverlap], allRadials[bestOverlap]))

                        job2Object[otherIndex] = -1
                        jobsToRemove.append(self.prevJobs[otherIndex])
                        job2Object.append(bestOverlap)
                    else :
                        # The other job had better match, so end this one instead
                        job2Object.append(-1)
                        jobsToRemove.append(oldJob)
                else :
                    oldJob.reset(ChunkIter(gridshape, widths[bestOverlap], allRadials[bestOverlap]))
                    job2Object.append(bestOverlap)

        #print "Rad Counts:", radCnts
        jobsToAdd = []
        slicesToAdd = []
        for index, (radials, cnt, width) in enumerate(zip(allRadials, radCnts, widths)) :
            if cnt >= 20 and not index in job2Object :
                # An object without a pre-existing job!
                jobsToAdd.append(StaticJob(timedelta(seconds=40),
                               #(radials,),
                               ChunkIter(gridshape, width, radials),
                               dwellTime=timedelta(microseconds=64000),
                               prt=timedelta(microseconds=800)))
                slicesToAdd.append(objects[index])

        jobsToKeep = [aJob for aJob in self.prevJobs if aJob not in jobsToRemove]
        slicesToKeep = [aSlice for aJob, aSlice in 
                        zip(self.prevJobs, self._jobRegions) if
                        aJob not in jobsToRemove]

        self.prevJobs = jobsToKeep + jobsToAdd
        self._jobRegions = slicesToKeep + slicesToAdd
        return jobsToAdd, jobsToRemove
register_sensing(SimpleTrackingSys)


