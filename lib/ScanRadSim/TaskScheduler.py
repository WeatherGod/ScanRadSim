import numpy as np
from datetime import timedelta

def _to_usecs(somedelta) :
    return (86400000000 * somedelta.days) + (1000000 * somedelta.seconds) + somedelta.microseconds

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

    def _remain_time(self, job) :
        """
        This function is to return the remaining time for an
        active task for a particular job.

        This function has an important purpose:
        If a job has a currently active task, then its loopcnt
        will see that entire task as complete.  But that
        completion includes things still to be done.

        Let's illustrate this with an example.  Suppose we
        query the scheduler every second, and at one point,
        we set active a task that takes 5 seconds to complete.
        For the next 4 queries, that job will have a loopcnt
        that says that it has completed the entire task,
        but that hasn't happened yet.

        In such a situation, calculating the true_update_period()
        will be wrong if the time given to it is only the time
        "so far".  Therefore, we correct this by providing this
        function which checks to see if the job has an active
        task and returns the time remaining for the task.
        """
        if job is not None and job.is_running :
            for index in range(len(self.active_tasks)) :
                if job.currtask is self.active_tasks[index] :
                    return job.currtask.T - self._active_time[index]

            raise ValueError("Nothing matched!  Maybe this job's task belonged to another scheduler?")
        else :
            return timedelta(0)

    # NOTE: These next few functions are temporarially assuming the existance
    #       of a member variable called "self.surveil_job".
    def occupancy(self) :
        Ts, lens, Us = zip(*[(aJob.T, len(aJob._origradials),
                              aJob.true_update_period(self._remain_time(aJob) + jobtime)) for
                             aJob, jobtime in zip(self.jobs + [self.surveil_job],
                                                  self._job_lifetimes + [self._schedlifetime])])
        try :
            return sum([(_to_secs(t * aLen) / float(_to_secs(u))) for
                        t, aLen, u in zip(Ts, lens, Us) if
                        t != timedelta() and u != timedelta.max]) / self._concurrent_max
        except ValueError :
            # In the case there are no valid values to sum
            return np.nan

    def acquisition(self) :

        U_times = [aJob.true_update_period(self._remain_time(aJob) + jobtime) for
                   aJob, jobtime in
                   zip(self.jobs + [self.surveil_job],
                       self._job_lifetimes + [self._schedlifetime])]
        #print U_times
        #print [str(life) for life in self._job_lifetimes + [self._schedlifetime]]

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
        #print [self._remain_time(aJob) for aJob in self.jobs]
        U_times = [aJob.true_update_period(self._remain_time(aJob) + joblife) for
                   aJob, joblife in zip(self.jobs, self._job_lifetimes)]
        if len(U_times) > 0 :
            return (sum([1.0 / _to_secs(u) for u in
                         U_times if u != timedelta.max]) *
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

    def next_jobs(self, auto_activate=False) :
        raise NotImplementedError("next_jobs() needs to be implemented by the derived class!")

    def add_active(self, theJob, auto_activate=False) :
        for index, activeTask in enumerate(self.active_tasks) :
            if activeTask is None :
                theTask = theJob.next()
                # This gets changed to True by the scan simulator,
                # because that is when the scan is actually active.
                # Or auto_activate can be set to True.
                # Note that ScanSim checks to see if the task is
                # already running before using it, and will skip
                # it if it is running already.
                theTask.is_running = auto_activate
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

