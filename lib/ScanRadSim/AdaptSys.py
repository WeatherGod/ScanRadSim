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
        
        jobsToAdd = [StaticJob(timedelta(seconds=40),
                               ChunkIter(gridshape, width, radials),
                               dwellTime=timedelta(microseconds=64000),
                               prt=timedelta(microseconds=800)) for
                     radials, cnt, width in zip(allRadials, radCnts, widths)]

        self.prevJobs = jobsToAdd
        return jobsToAdd, jobsToRemove
register_sensing(SimpleSensingSys)



class PPISensingSys(AdaptSenseSys) :
    """
    Just scan for every contiguous +35dBz region, but in a PPI fashion.
    """
    name = "SimplePPI"
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
register_sensing(PPISensingSys)


