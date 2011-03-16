from BRadar.io import LoadLevel2
from RadarInterpolator import interp_radar
from task import Task


class Simulator(object) :
    def __init__(self, files, theSys, sched) :
        if len(files) < 2 :
            raise(ValueError, "Need at least 2 files for a simulation")

        radData = [LoadLevel2(aFile) for aFile in files]

        # Each element in this list is a 3-D array (elev, azi, range)
        self.radData = [aTime['vals'] for aTime in radData]
        self.times = [aTime['scan_time'] for aTime in radData]

        self.aziGrid = radData[0]['azimuth']
        self.rngGrid = radData[0]['range_gate']
        self.elvGrid = radData[0]['elev_angle']

        self.adaptSense = theSys
        self.scheduler = sched
        self.currIndex = 0
        self.currView = self.radData[self.currIndex].copy()

    def next(self, timeElapsed) :
        newTime = self.times[self.currIndex] + timeElapsed
        if newTime >= self.times[self.currIndex + 1] :
            # We are onto the next file.
            self.currIndex += 1
            self.currView = self.radData[self.currIndex].copy()

        if currIndex >= len(self.times) :
            return



        newtasks, fintasks = self.adaptSense(self.currView)

        self.scheduler.remove_tasks(fintasks)
        self.scheduler.add_tasks(newtasks)
        doTask = self.scheduler.next_task()


        newData = interp_radar(self.radData[currIndex],
                               self.radData[currIndex + 1],
                               [float((newTime - self.times[self.currIndex]).microseconds) /
                                (self.times[self.currIndex + 1] -
                                 self.times[self.currIndex]).microseconds])
        taskRadials = doTask.next()
        self.currView[taskRadials, :] = newData[taskRadials, :]

        # Return the amount of time to wait until the next schedule
        return doTask.T


from scipy.ndimage.measurements import find_objects, label
class SimpleSensingSys(object) :
    """
    Just scan for every contiguous +35dBz region.
    """
    def __init__(self) :
        self.prevTasks = []

    def __call__(self, radData) :
        tasksToRemove = self.prevTasks

        labels, cnt = label(radData)
        objects = find_objects(labels)
        radialCnts = [(np.mgrid[anObject[:2]]).size // 2 for
                      anObject in objects]
        tasksToAdd = [Task(datetime.timedelta(seconds=2),
                           datetime.timedelta(microseconds=10 * radCnt),
                           anObject[:2]) for
                      anObject, radCnt in zip(objects, radialCnts)]
        
        self.prevTasks = tasksToAdd
        return tasksToAdd, tasksToRemove
