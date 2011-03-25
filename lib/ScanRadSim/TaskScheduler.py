import numpy as np
from datetime import timedelta

def _to_usecs(somedelta) :
    return (86400000000 * somedelta.days) + (60000000 * somedelta.seconds) + somedelta.microseconds

def _to_secs(somedelta) :
    return 1e-6 * _to_usecs(somedelta)

class TaskScheduler(object) :
    """
    Base class for any radar task scheduler.
    """
    def __init__(self, concurrent_max=1) :
        assert(concurrent_max >= 1)
        self.active_tasks = [None] * concurrent_max
        self._active_time = [None] * concurrent_max
        self.jobs = []
        self._job_lifetimes = []
        self._concurrent_max = concurrent_max

        # This is just used for some internal book-keeping.
        # Do not depend on this as a model timestamp.
        self._schedlifetime = timedelta()

        # These are here just to help determine how efficiently
        # we are incrementing the schedule's timer.
        self.max_timeOver = timedelta()
        self.sum_timeOver = timedelta()

    # NOTE: These next few functions are temporarially assuming the existance
    #       of a member variable called "self.surveil_job".
    def occupancy(self) :
        return sum([(_to_usecs(aJob.T * len(aJob._origradials)) /
                     float(_to_usecs(aJob.true_update_period(jobtime)))) for
                    aJob, jobtime in zip(self.jobs + [self.surveil_job],
                                         self._job_lifetimes + [self._schedlifetime]) if
                    _to_usecs(aJob.T) != 0]) / self._concurrent_max

    def acquisition(self) :
        U_times = [aJob.true_update_period(jobtime) for aJob, jobtime in
                   zip(self.jobs + [self.surveil_job],
                       self._job_lifetimes + [self._schedlifetime])]

        try :
            max_u = _to_secs(max([prd for prd in U_times if prd != timedelta.max]))
            return sum([max_u * _to_secs(aJob.T * len(aJob._origradials)) /
                        _to_secs(prd) for aJob, prd in
                        zip(self.jobs + [self.surveil_job], U_times) if
                        prd != timedelta.max])
        except ValueError :
            return np.nan

    def improve_factor(self, base_update_period) :
        """
        Calculate the improvement factor for the scheduling algorithm compared to
        a scan performed by a single beam in a non-adaptive manner (i.e., conventional
        WSR-88D scanning).
        In other words, the improvement factor is the average number of scans divided
        by the number of scans that would have been performed by a single radar beam.
        """
        if len(self.jobs) > 0 :
            return (sum([1.0 / _to_secs(aJob.true_update_period(joblife)) for
                         aJob, joblife in zip(self.jobs, self._job_lifetimes)]) *
                    _to_secs(base_update_period) / len(self.jobs))
        else :
            return np.nan

    def increment_timer(self, timeElapsed) :
        self._schedlifetime += timeElapsed

        for index in range(len(self._job_lifetimes)) :
            self._job_lifetimes[index] += timeElapsed

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
        self._job_lifetimes.extend([timedelta() for
                                    index in range(len(jobs))])

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

        # Need to go through these indices in reverse order
        # as we delete items from a list.  This way, the
        # subsequent index values are still valid.
        args = np.argsort(findargs)[::-1]
        for anItem in args :
            del self.jobs[findargs[anItem]]
            del self._job_lifetimes[findargs[anItem]]

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

