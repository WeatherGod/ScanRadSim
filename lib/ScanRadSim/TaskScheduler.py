import numpy as np
from datetime import timedelta


class TaskScheduler(object) :
    """
    Base class for any radar task scheduler.
    """
    def __init__(self, concurrent_max=1) :
        assert(concurrent_max >= 1)
        self.active_tasks = [None] * concurrent_max
        self._active_time = [None] * concurrent_max
        self.jobs = []

        self.max_timeOver = timedelta()
        self.sum_timeOver = timedelta()

    def increment_timer(self, timeElapsed) :
        for index, aTask in enumerate(self.active_tasks) :
            if aTask is not None :
                self._active_time[index] += timeElapsed

        self.rm_deactive()

    def is_available(self) :
        """
        Does the system have an available slot for a task execution?
        """
        return any([(activeTask is None) for
                    activeTask in self.active_tasks])

    def add_jobs(self, jobs) :
        self.jobs.extend(jobs)

    def rm_jobs(self, jobs) :
        # Slate these jobs for removal.
        # Note that you can't remove active operations until they are done.
        # We will go ahead and remove any jobs with inactive operation, and defer
        # the removal of jobs with active operations until later.
        # Impementation Note: This actually isn't all that complicated,
        #                     due to the reference counting of python.
        #                     Just delete the job from the self.jobs
        #                     list and when the job is done in the
        #                     active list, it will finally be deleted.
        findargs = [self.jobs.index(aJob) for aJob in jobs]
        args = np.argsort(findargs)[::-1]
        for anItem in args :
            del self.jobs[findargs[anItem]]

        return findargs, args

    def next_jobs(self) :
        raise NotImplementedError("next_jobs() needs to be implemented by the derived class!")

    def add_active(self, theJob) :
        for index, activeTask in enumerate(self.active_tasks) :
            if activeTask is None :
                theTask = theJob.next()
                # This gets changed to True by the scan simulator,
                # because that is when the scan is actually active.
                theTask.is_running = False
                self.active_tasks[index] = theTask
                self._active_time[index] = timedelta()
                return

        raise ValueError("FATAL: There were no available slots for this task!")

    def rm_deactive(self) :
        for index, (aTask, actTime) in enumerate(zip(self.active_tasks,
                                                     self._active_time)) :
            if aTask is not None :
                if actTime >= aTask.T :
                    # The task is finished its fragment!
                    aTask.is_running = False
                    timeDiff = actTime - aTask.T
                    self.max_timeOver = max(self.max_timeOver, timeDiff)
                    self.sum_timeOver += timeDiff
                    self._active_time[index] = None
                    self.active_tasks[index] = None

