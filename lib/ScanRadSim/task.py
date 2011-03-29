from itertools import cycle, tee
import datetime
import numpy as np

from NDIter import SliceIter, BaseNDIter

def _slicesize(theSlice) :
    try :
        return int(np.prod([len(range(aSlice.start, aSlice.stop,
                                      1 if aSlice.step is None else aSlice.step)) for
                            aSlice in theSlice]))
    except :
        # This might be wrong...
        return len(theSlice)

class ScanOperation(object) :
    def __init__(self, job, radSlice, tx_time, rx_time, wait_time=None) :
        """
        Times for the three parts of any scan operation using timedelta objects.
        tx == transmit
        rx == receive

        A Scan Operation can not be pre-empted during the transmit and receive modes.
        """
        self.job = job
        self.tx_time = tx_time
        self.rx_time = rx_time
        self.wait_time = wait_time
        self.currslice = radSlice
        self.T = tx_time + rx_time
        # This should be set to True by the scanner
        # and then set to False when its operation is complete.
        self.is_running = False
        if wait_time is not None :
            self.T += wait_time

    def _slicesize(self) :
        return _slicesize(self.currslice[:-1])

class ScanJob(object) :
    def __init__(self, radials, doCycle=False) :
        """
        radials is any iterator that returns an object that
            can be used to access a part of a numpy array upon
            a call to next().

        doCycle will indicate whether or not to cycle through
        the radials.  Default is False.
        """
        # This is so I can still access the info in radials,
        # regardless of whether or not it gets wrapped by
        # a cycle object.
        self._origradials = radials
        self._startingPoint, self.radials = tee(radials, 2)
        if doCycle :
            self.radials = cycle(self.radials)
        #self.currslice = None
        #self.currtask = None
        self._nextcallCnt = 0
        self._recent_task = None

    """
    def _set_running(self, is_run) :
        if self.currtask is not None :
            self.currtask.is_running = is_run
        elif is_run :
            # Only worry about it if trying to set it to True.
            raise ValueError("This scanjob does not have an active task yet!")

    def _get_running(self) :
        if self.currtask is not None :
            return self.currtask.is_running
        else :
            # If there is no current task, then it isn't running, right?
            return False

    is_running = property(_get_running, _set_running, None, "The status of the scan-job's current task")
    """

    def _timeForJob(self) :
        """
        Default method of determining how much time the job takes.
        Will not work for all radials iterators, so any special situations
        must over-ride this function.

        For example, any radials iterator derived from BaseNDIter will 
        have issues because running this function will cause the iterator's
        internal mechanisms to move forward.  Some other functions may depend
        on these mechanisms and will be mislead, in particular the VCP class.
        """
        tempIter, = tee(self._startingPoint, 1)
        timeToComplete = datetime.timedelta(0)
        for aSlice in tempIter :
            timeToComplete += self._timeToComplete(aSlice[:-1])
        return timeToComplete

    def _timeToComplete(self, thisSlice) :
        return self.dwellTime * _slicesize(thisSlice)

    def _loopcnt(self) :
        chunkCnt = len(self._origradials)
        if chunkCnt != 0 :
            return int(self._nextcallCnt // chunkCnt)
        else :
            return 0

    loopcnt = property(_loopcnt, None, None, "Find out how many cycles the radials iterator has made")

    def _loopcnt_frac(self) :
        chunkCnt = len(self._origradials)
        if chunkCnt != 0 :
            return self._nextcallCnt / float(chunkCnt)
        else :
            return 0.0

    loopcnt_frac = property(_loopcnt_frac, None, None,
                            "Find out how many cycles (and the fractional amount of the current loop) the radials iterator has made")

    def true_update_period(self, elapsedTime) :
        """
        Figure out what the actual update period has been for this scan job,
        given that the amound of time that has elapsed is given.
        """
        # The fractions module is needed to produce a more accurate value for the
        # update period.  In the python 2.x series, float division of a timedelta
        # object is not allowed.
        # The following is only valid for python 2.6.
        from fractions import Fraction
        loop_frac = Fraction.from_float(self.loopcnt_frac).limit_denominator(100)  # 100 should be enough for everybody!
        #print self.loopcnt_frac, loop_frac
        if loop_frac.numerator != 0 :
            return (elapsedTime * loop_frac.denominator) / loop_frac.numerator
        else :
            return datetime.timedelta.max

    def __iter__(self) :
        return self

    def next(self) :
        self._nextcallCnt += 1
        # TODO: Assume a 10% duty cycle for now...
        #print self, self._origradials, self._origradials._chunkIndices, self._origradials._chunkCnts
        nextslice = self.radials.next()
        T = self._timeToComplete(nextslice[:-1])
        #print "Time to complete Task:", T
        txTime = T / 10
        rxTime = T - txTime

        #print "Scan Job:", self, "  T:", self.T, "  rad cnt:", self._slicesize()
        #print "Slice:", [aSlice.indices(aSlice.stop - aSlice.start) for aSlice in self.currslice],\
        #      "  T:", self.T, "  rad cnt:", self._slicesize(), " indices:", self.radials._chunkIndices
        self._recent_task = ScanOperation(self, nextslice, txTime, rxTime)
        return self._recent_task


class StaticJob(ScanJob) :
    def __init__(self, updatePeriod, radials, dwellTime, prt=None, doCycle=True) :
        """
        updatePeriod must be a timedelta from datetime module and represents the
            period of time between cycles of the radials iterator.
        dwellTime and prt are also timedelta objects
            dwellTime is the time it takes to process *a* radial and is assumed
            to be constant throughout the scan job.
        radials will be any iterator that returns an item that can be used to
            access a part or sector of a numpy array upon a call to next()
        """
        if prt is None :
            # For now, assume 10 pulses
            prt = dwellTime / 10

        ScanJob.__init__(self, radials, doCycle)

        self.dwellTime = dwellTime
        self.prt = prt
        self.T = self._timeForJob()
        self.U = max(updatePeriod, self.T)


WSR_88D_PRT =  {1:  datetime.timedelta(microseconds=int(round(1e6 / 322))),
                2:  datetime.timedelta(microseconds=int(round(1e6 / 446))),
                3:  datetime.timedelta(microseconds=int(round(1e6 / 644))),
                4:  datetime.timedelta(microseconds=int(round(1e6 / 857))),
                5:  datetime.timedelta(microseconds=int(round(1e6 / 1014))),
                6:  datetime.timedelta(microseconds=int(round(1e6 / 1095))),
                7:  datetime.timedelta(microseconds=int(round(1e6 / 1181))),
                8:  datetime.timedelta(microseconds=int(round(1e6 / 1282)))}

WSR_88D_Cuts = {21 : (0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8),
                12 : (0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
                11 : (0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
               121 : (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8),
                31 : (0, 0, 1, 1, 2, 2, 3, 4),
                32 : (0, 0, 1, 1, 2, 3, 4)}

WSR_88D_Elevs = {21 : (0.5, 1.45, 2.4, 3.35, 4.3, 6.0, 9.0, 14.6, 19.5),
                 12 : (0.5, 0.9, 1.3, 1.8, 2.4, 3.1, 4.0, 5.1, 6.4, 8.0, 10.0, 12.5, 15.6, 19.5),
                 11 : (0.5, 1.45, 2.4, 3.35, 4.3, 5.25, 6.2, 7.5, 8.7, 10.0, 12.0, 14.0, 16.7, 19.5),
                121 : (0.5, 1.45, 2.4, 3.35, 4.3, 6.0, 9.9, 14.6, 19.5),
                 31 : (0.5, 1.5, 2.5, 3.5, 4.5),
                 32 : (0.5, 1.5, 2.5, 3.5, 4.5)}

WSR_88D_PRT_Num = {21 : (1, 5, 1, 5, (2, 5), (2, 5), (2, 5), (3, 5), 7, 7, 7),
                   12 : (1, 5, 1, 5, 1, 5, (1, 5), (2, 5), (2, 5), (2, 5), (3, 5), (3, 5), 6, 7, 8, 8, 8),
                   11 : (1, 5, 1, 5, (1, 5), (2, 5), (2, 5), (3, 5), (3, 5), 6, 7, 7, 7, 7, 7, 7),
                  121 : (1, 8, 6, 4, 1, 8, 6, 4, (1, 8), 6, 4, (2, 8), 6, 4, (2, 4), 7, (3, 5), 7, 8, 8),
                   31 : (1, 2, 1, 2, 1, 2, 2, 2),
                   32 : (1, 5, 1, 5, (2, 5), (2, 5), (2, 5))}

WSR_88D_PlsCnts = {21 : (28, 88, 28, 88, (8, 70), (8, 70), (8, 70), (12, 70), 82, 82, 82),
                   12 : (15, 40, 15, 40, 15, 40, (3, 40), (3, 29), (3, 30), (3, 30), (3, 30), (3, 30), (3, 30), 38, 40, 44, 44, 44),
                   11 : (17, 52, 16, 52, (6, 41), (6, 41), (6, 41), (10, 41), (10, 41), 43, 46, 46, 46, 46, 46, 46),
                  121 : (11, 43, 40, 40, 11, 43, 40, 40, (6, 40), 40, 40, (6, 40), 40, 40, (6, 40), 40, 43, 43),
                   31 : (63, 87, 63, 87, 63, 87, 87, 87),
                   32 : (64, 220, 64, 220, (11, 220), (11, 220), (11, 220))}

def _wsr_dwelltime(vcp) :
    dwells = []
    for index, cnt in zip(WSR_88D_PRT_Num[vcp],
                          WSR_88D_PlsCnts[vcp]) :
        tot = datetime.timedelta()
        # Cast everything into tuples because of batch modes,
        # which require summing in order to get a correct dwell time.
        if not isinstance(index, tuple) :
            index = (index,)
            cnt = (cnt,)

        for pulseCnt, indx in zip(cnt, index) :
            tot += (pulseCnt * WSR_88D_PRT[indx])
        dwells.append(tot)
    return dwells

def _wsr_prts(vcp) :
    """
    This isn't always *real* prts, just average prts in case of batch mode.
    """
    prts = []
    for dwell, cnt in zip(_wsr_dwelltime(vcp),
                          WSR_88D_PlsCnts[vcp]) :
        if not isinstance(cnt, tuple) :
            cnt = (cnt,)

        prts.append(dwell / sum(cnt))
    return prts

class VCP(ScanJob) :
    def __init__(self, vcp, gridshape, slices=None, elevOffset=0, updatePeriod=None, doCycle=True) :
        """
        A scan job that mimics the scanning pattern and timing
        of a specified WSR-88D VCP.

        If the entire grid is actually a raised slice, then I need
           to know how much it is raised so that I can access the
           proper information at that elevation level in the absolute
           coordinate system.
            Note, this has nothing to do with "slices" representing
            a subset of the elevation angles. Only that the overall grid
            that this job operates within is raised.

        Update period is a timedelta object that specifies the update
            period that this job should be set to.  Note that if it
            is too small, it will be adjusted to the time it takes
            to complete one run of the VCP.
            If None, then use the default WSR-88D update time for
            the time it takes to cover "gridshape".
        """
        if slices is None :
            slices = [slice(None) for shape in gridshape]

        chunkSize = 5

        dwellTimes_elevs = _wsr_dwelltime(vcp)
        prts_elevs = _wsr_prts(vcp)

        ## This must be done before remaking gridshape because I
        ## need to know how wide the original grid was.
        #if updatePeriod is None :
        #    updatePeriod = datetime.timedelta()
        #    for aTime in dwellTimes_elevs :
        #        updatePeriod += aTime * gridshape[1]

        # These grid values are in the grid coordinate system
        # (which is relative to the absolute coordinates by elevOffset),
        # but has to be derived from the size of the data grid.
        cutlist = range(*slices[0].indices(gridshape[0]))

        # Remaking gridshape so that the slices and shapes agree.                
        self._gridshape = [len(range(*aSlice.indices(shape))) for
                           aSlice, shape in zip(slices, gridshape)]

        # The elevation indices in slice coordinate system,
        # (which is relative to the grid coordinate system by slice.start,
        # (which, in turn, is relative to the abs coordinate system by elevOffset))
        # in the order of execution.
        sliceOffset = min(cutlist)
        elevs, self._dwellTimes, self._prts = \
                zip(*[(elev - elevOffset - sliceOffset, dwell, prt) for
                      elev, dwell, prt in
                      zip(WSR_88D_Cuts[vcp], dwellTimes_elevs, prts_elevs) if
                      (elev - elevOffset in cutlist)])

        #print slices[1], gridshape[1]
        # The extra element is so that we can iterate all the way through.
        azidivs = range(0, self._gridshape[1], chunkSize) + [self._gridshape[1]]

        chunkIters = [cycle(slice(start, start + 1) for
                            start in elevs),
                      cycle(slice(start, stop, 1) for start, stop
                            in zip(azidivs[:-1], azidivs[1:])),
                      cycle((slice(0, self._gridshape[2], 1),))]
        chunkCnts = [len(elevs), len(azidivs) - 1, 1]

        #print self, "Azidivs:", azidivs
        iterChunk = BaseNDIter(chunkIters, chunkCnts, (1, 0, 2), doCycle)

        ScanJob.__init__(self, iterChunk, doCycle=False)
        self.T = self._timeForJob()
        self.U = max(updatePeriod if updatePeriod is not None else datetime.timedelta(0),
                     self.T)

    def _timeForJob(self) :
        timeToComplete = datetime.timedelta(0)
        for dwell in self._dwellTimes :
            timeToComplete += (dwell * self._gridshape[1])
        return timeToComplete

    def _get_dwelltime(self) :
        # Based on the current elevation angle,
        # return the dwell time.
        if self._origradials._started :
            return self._dwellTimes[self._origradials._chunkIndices[0]]
        else :
            return datetime.timedelta()

    dwellTime = property(_get_dwelltime, None, None, "The current dwell time")
        
    def _get_prt(self) :
        # Based on the current elevation angle,
        # return the prt
        if self._origradials._started :
            return self._prts[self._origradials._chunkIndices[0]]
        else :
            return datetime.timedelta()

    prt = property(_get_prt, None, None, "The current prt")


class Surveillance(ScanJob) :
    def __init__(self, dwellTime, gridshape, slices=None, prt=None, doCycle=True) :
        """
        dwellTime and prt must be integers in units of microseconds.
        gridshape must be a tuple of ints representing the shape of the
            *entire* radar grid.
        slices is a tuple that represents the portion of the grid
            this surveillance job is responsible for. If None, then
            assume the entire grid.
        """
        if slices is None :
            slices = [slice(0, shape, 1) for shape in gridshape]

        chunkSize = 5

        gridshape = [len(range(*aSlice.indices(shape))) for
                     aSlice, shape in zip(slices, gridshape)]

        # Get the slice-chunking iterator for this task.
        #print "start:", [0] * len(gridshape), " stop:", gridshape
        iterChunk = SliceIter([0] * len(gridshape), gridshape,
                              [1, chunkSize, gridshape[-1]],
                              (1, 0, 2))

        radialCnt = int(np.prod(gridshape[:-1]))
        dwellTime = datetime.timedelta(microseconds=dwellTime)
        updatePeriod = dwellTime * radialCnt
        #print "Dwell:", dwellTime, "  Radials:", radialCnt, "  ChunkCnt:", len(iterChunk)

        ScanJob.__init__(self, iterChunk, doCycle)
        self.U = updatePeriod
        self.T = updatePeriod
        self.prt = prt
        self.dwellTime = dwellTime




if __name__ == '__main__' :
    gridshape = (9, 92, 1000)
    vol = (slice(0, 9), slice(0, 92, 1), slice(None))

    print "Surveillance Job"
    print "-----------------"
    a = Surveillance(64000, gridshape, slices=vol, doCycle=False)

    print a.radials, len(a._origradials), a.U
    
    for index, b in zip(range(20), a) :
        #print a.radials._chunkIndices, a.radials._chunkCnts
        print b.T, b.currslice

    print "\n"
    print "VCP 21 Job"
    print "----------"
    a = VCP(21, gridshape, slices=vol, doCycle=False, elevOffset=0)

    print a.radials, len(a._origradials), a.U

    for b in a :
        print b.T, b.currslice

    print "\nVCP 21 Job"
    print "----------"
    a = VCP(21, gridshape, slices=vol, doCycle=False, elevOffset=1)

    print a.radials, len(a._origradials), a.U

    for index, b in zip(range(40), a) :
        print b.T, b.currslice

    print "\nVCP 21 Job"
    print "----------"
    vol = (slice(0, 3), slice(20, 184, 1), slice(None))
    a = VCP(21, gridshape, slices=vol, doCycle=False, elevOffset=0)

    print a.radials, len(a._origradials), a.U

    for index, b in zip(range(40), a) :
        print b.T, b.currslice
