from BRadar.io import LoadLevel2
#from RadarInterpolator import interp_radar
from task import Task
import numpy as np


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
                       float((self.nextItem['scan_time'] -
                              self.currItem['scan_time']).microseconds))

    def update(self, timeElapsed, taskRadials) :
        # Don't do += because we don't want to modify the original
        # 'scan_time' value in the first self.currItem
        self.currTime = self.currTime + timeElapsed
        if self.currTime >= self.nextItem['scan_time'] :
            # We move onto the next file.
            self.currItem = self.nextItem
            self.nextItem = self.radData.next()

            if self.nextItem is None :
                return

            self._set_slope()

        self.currView[taskRadials, :] = (self._slope[taskRadials, :] *
                                         (self.currTime - self.currItem['scan_time']).microseconds)



from scipy.ndimage.measurements import find_objects, label
class SimpleSensingSys(object) :
    """
    Just scan for every contiguous +35dBz region.
    """
    def __init__(self) :
        self.prevTasks = []

    def __call__(self, radData) :
        tasksToRemove = self.prevTasks

        labels, cnt = label(radData >= 35.0)
        objects = find_objects(labels)
        radialCnts = [(np.mgrid[anObject[:2]]).size // 2 for
                      anObject in objects]
        tasksToAdd = [Task(datetime.timedelta(seconds=2),
                           datetime.timedelta(microseconds=10 * radCnt),
                           anObject[:2]) for
                      anObject, radCnt in zip(objects, radialCnts)]
        
        self.prevTasks = tasksToAdd
        return tasksToAdd, tasksToRemove
