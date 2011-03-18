from BRadar.io import LoadLevel2
#from RadarInterpolator import interp_radar
from task import Task
import numpy as np
from datetime import timedelta


class Simulator(object) :
    def __init__(self, files) :
        self.radData = (LoadLevel2(aFile) for aFile in files)

        self.currItem = self.radData.next()
        self.nextItem = self.radData.next()

        if self.nextItem is None :
            raise(ValueError, "Need at least 2 files for a simulation")


        self.currTime = self.currItem['scan_time']
        self.currView = np.empty_like(self.currItem['vals'])
        self.currView.fill(np.nan)

        self._set_slope()


    def _set_slope(self) :
        self._slope = ((self.nextItem['vals'] - self.currItem['vals']) /
                       self._time_diff(self.currItem['scan_time'],
                                       self.nextItem['scan_time']))

    def _time_diff(self, time1, time2) :
        """
        Return the time difference in units of seconds,
        including the microsecond portion.
        """
        timediff = time2 - time1
        return timediff.seconds + (timediff.microseconds * 1e-6)

    def update(self, theTask) :
        timeElapsed = theTask.T
        self.currTime += timeElapsed
        if self.currTime >= self.nextItem['scan_time'] :
            # We move onto the next file.
            self.currItem = self.nextItem

            try :
                self.nextItem = self.radData.next()
            except StopIteration :
                return False
            
            self._set_slope()

        taskRadials = theTask.next()
        #print type(taskRadials)
        self.currView[taskRadials] = ((self._slope[taskRadials] *
                                       self._time_diff(self.currItem['scan_time'],
                                                       self.currTime)) +
                                      self.currItem['vals'][taskRadials])

        return True


from scipy.ndimage.measurements import find_objects, label
class SimpleSensingSys(object) :
    """
    Just scan for every contiguous +35dBz region.
    """
    def __init__(self) :
        self.prevTasks = []

    def __call__(self, radData) :
        tasksToRemove = self.prevTasks

        # The label and find_objects will slice only the
        # relevant radials, but we still need something to
        # represent the entire length of the radial.
        fullRange = slice(None, None, None)

        # Find the maximum value along each radial.
        maxView = np.nanmax(radData, axis=-1)
        labels, cnt = label(maxView >= 35.0)
        objects = find_objects(labels)

        allRadials = [radials + (fullRange,) for radials in objects]
        radCnts = [int(np.prod([aSlice.stop - aSlice.start for
                            aSlice in radials])) for
                   radials in objects]
        

        tasksToAdd = [Task(timedelta(seconds=20),
                           timedelta(milliseconds=10*cnt),
                           radials, prt=10000) for
                      radials, cnt in zip(allRadials, radCnts) if
                      cnt >= 20]
        
        self.prevTasks = tasksToAdd
        return tasksToAdd, tasksToRemove
