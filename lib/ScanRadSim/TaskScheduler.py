
from datetime import timedelta


class TaskScheduler(object) :
    """
    Base class for any radar task scheduler.
    """
    def __init__(self, concurrent_max=1) :
        assert(concurrent_max >= 1)
        self.active_tasks = [None] * concurrent_max
        self._active_time = [None] * concurrent_max

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

    def next_task(self) :
        raise NotImplementedError("next_task() needs to be implemented by the derived class!")

    def add_active(self, theTask) :
        for index, activeTask in enumerate(self.active_tasks) :
            if activeTask is None :
                theTask.is_running = True
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
                    self._active_time[index] = None
                    self.active_tasks[index] = None

