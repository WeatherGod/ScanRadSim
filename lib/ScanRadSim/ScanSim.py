from BRadar.io import LoadLevel2
#from RadarInterpolator import interp_radar
import numpy as np
from datetime import timedelta, datetime


def _to_seconds(timediff) :
    """
    Takes the timedelta object and return the
    floating point value of the time difference
    in seconds.
    """
    return (86400.0 * timediff.days) + timediff.seconds + (1e-6 * timediff.microseconds)

class Simulator(object) :
    def __init__(self, files) :
        self.radData = (LoadLevel2(aFile) for aFile in files)

        self.currItem = self.radData.next()
        self.nextItem = self.radData.next()

        if self.nextItem is None :
            raise(ValueError, "Need at least 2 files for a simulation")


        self.currView = np.empty_like(self.currItem['vals'])
        self.currView.fill(np.nan)
        self.radialAge = np.empty(self.currItem['vals'].shape[:-1],
                                  dtype=datetime)
        self.radialAge.fill(self.currItem['scan_time'])
        self.updateCnt = np.zeros(self.currItem['vals'].shape[:-1], dtype=np.int)

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
        return _to_seconds(time2 - time1)

    def update(self, theTime, theTasks, volume=None) :
        if volume is None :
            volume = (slice(None),slice(None),slice(None))

        if theTime >= self.nextItem['scan_time'] :
            # We move onto the next file.
            self.currItem = self.nextItem

            try :
                self.nextItem = self.radData.next()
            except StopIteration :
                return False
            
            self._set_slope()

        for aTask in theTasks :
            if aTask is None or aTask.is_running :
                continue

            aTask.is_running = True
            taskRadials = aTask.currslice
            #print aTask, taskRadials
            self.currView[volume][taskRadials] = ((self._slope[volume][taskRadials] *
                                           self._time_diff(self.currItem['scan_time'],
                                                           theTime)) +
                                          self.currItem['vals'][volume][taskRadials])
            # Reset the age of these radials.
            self.radialAge[volume[:-1]][taskRadials[:-1]] = theTime
            self.updateCnt[volume[:-1]][taskRadials[:-1]] += 1

        return True


