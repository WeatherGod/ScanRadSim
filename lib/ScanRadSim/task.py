from itertools import cycle
import datetime
import numpy as np

from NDIter import SliceIter, BaseNDIter

class ScanOperation(object) :
    def __init__(self, radSlice, tx_time, rx_time, wait_time=None) :
        """
        Times for the three parts of any scan operation using timedelta objects.
        tx == transmit
        rx == receive

        A Scan Operation can not be pre-empted during the transmit and receive modes.
        """
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


class ScanJob(object) :
    def __init__(self, radials, doCycle=False) :
        """
        radials is any iterator that returns an object that
            can be used to access a part of a numpy array upon
            a call to next().

        doCycle will indicate whether or not to cycle through
        the radials.  Default is False.
        """
        self.is_running = False
        # This is so I can still access the info in radials,
        # regardless of whether or not it gets wrapped by
        # a cycle object.
        self._origradials = radials
        self.radials = cycle(radials) if doCycle else radials
        self.currslice = None

    def __iter__(self) :
        return self

    def _slicesize(self) :
        if self.currslice is not None :
            #print self.currslice
            return int(np.prod([len(range(aSlice.start, aSlice.stop)) for
                                aSlice in self.currslice[:-1]]))
        else :
            return 0

    def _timeToComplete(self) :
        return self.dwellTime * self._slicesize()

    def next(self) :
        # TODO: Assume a 10% duty cycle for now...
        #print self, self._origradials, self._origradials._chunkIndices, self._origradials._chunkCnts
        self.currslice = self.radials.next()
        self.T = self._timeToComplete()
        txTime = self.T / 10
        rxTime = self.T - txTime

        #print "Scan Job:", self, "  T:", self.T, "  rad cnt:", self._slicesize()
        #print "Slice:", [aSlice.indices(aSlice.stop - aSlice.start) for aSlice in self.currslice],\
        #      "  T:", self.T, "  rad cnt:", self._slicesize(), " indices:", self.radials._chunkIndices
        return ScanOperation(self.currslice, txTime, rxTime)

class StaticJob(ScanJob) :
    def __init__(self, updatePeriod, radials, dwellTime, prt=None, doCycle=True) :
        """
        updatePeriod must be a timedelta from datetime module and represents the
            period of time between calls to next() from the radials iterator.
        dwellTime and prt are also timedelta objects
            dwellTime is the time it takes to process the radials from a call
            to radials.next()
        radials will be any iterator that returns an item that can be used to
            access a part or sector of a numpy array upon a call to next()

        This class is different from ScanJob in that the updatePeriod
            is adjusted upward in case the time to complete the scan
            is greater than the given updatePeriod.
        """
        timeToComplete = len(radials) * dwellTime
        if updatePeriod < timeToComplete :
            updatePeriod = timeToComplete

        if prt is None :
            # For now, assume 10 pulses
            prt = dwellTime / 10

        ScanJob.__init__(self, radials, doCycle)
        self.U = updatePeriod / len(radials)
        self.prt = prt
        self.dwellTime = dwellTime
        self.T = datetime.timedelta()
        

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
            slices = [slice(0, shape, 1) for shape in gridshape]

        chunkSize = 5

        dwellTimes_elevs = _wsr_dwelltime(vcp)
        prts_elevs = _wsr_dwelltime(vcp)

        # This must be done before remaking gridshape because I
        # need to know how wide the original grid was.
        if updatePeriod is None :
            updatePeriod = datetime.timedelta()
            for aTime in dwellTimes_elevs :
                updatePeriod += aTime * gridshape[1]

        # These grid values are in the grid coordinate system
        # (which is relative to the absolute coordinates by elevOffset),
        # but has to be derived from the size of the data grid.
        cutlist = range(*slices[0].indices(gridshape[0]))

        # Remaking gridshape so that the slices and shapes agree.                
        gridshape = [len(range(*aSlice.indices(shape))) for
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
        azidivs = range(0, gridshape[1], chunkSize) + [gridshape[1]]

        chunkIters = [cycle(slice(start, start + 1) for
                            start in elevs),
                      cycle(slice(start, stop, 1) for start, stop
                            in zip(azidivs[:-1], azidivs[1:])),
                      cycle((slice(0, gridshape[2], 1),))]
        chunkCnts = [len(elevs), len(azidivs) - 1, 1]

        #print self, "Azidivs:", azidivs
        iterChunk = BaseNDIter(chunkIters, chunkCnts, (1, 0, 2))

        timeToComplete = datetime.timedelta()
        for elev_sliced in elevs :
            timeToComplete += (dwellTimes_elevs[elev_sliced + elevOffset] * gridshape[1])

        if timeToComplete > updatePeriod :
            updatePeriod = timeToComplete

        ScanJob.__init__(self, iterChunk, doCycle)
        self.U = updatePeriod# / int(np.prod(chunkCnts))
        self.T = datetime.timedelta()

    def _get_dwelltime(self) :
        # Based on the current elevation angle,
        # return the dwell time.
        if self.currslice is not None :
            return self._dwellTimes[self._origradials._chunkIndices[0]]
        else :
            return datetime.timedelta()

    dwellTime = property(_get_dwelltime, None, None, "The current dwell time")
        
    def _get_prt(self) :
        # Based on the current elevation angle,
        # return the prt
        if self.currslice is not None :
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
        self.U = updatePeriod# / len(iterChunk)
        self.T = datetime.timedelta()
        self.prt = prt
        self.dwellTime = dwellTime




if __name__ == '__main__' :
    gridshape = (9, 92, 1000)
    vol = (slice(0, 9), slice(0, 92, 1), slice(None))

    print "Surveillance Job"
    print "-----------------"
    a = Surveillance(64000, gridshape, slices=vol, doCycle=False)

    print a.radials, len(a.radials), a.U
    
    for index, b in zip(range(20), a) :
        #print a.radials._chunkIndices, a.radials._chunkCnts
        print b.T, b.currslice

    print "\n"
    print "VCP 21 Job"
    print "----------"
    a = VCP(21, gridshape, slices=vol, doCycle=False, elevOffset=0)

    print a.radials, len(a.radials), a.U

    for b in a :
        print b.T, b.currslice

    print "\nVCP 21 Job"
    print "----------"
    a = VCP(21, gridshape, slices=vol, doCycle=False, elevOffset=1)

    print a.radials, len(a.radials), a.U

    for index, b in zip(range(40), a) :
        print b.T, b.currslice

    print "\nVCP 21 Job"
    print "----------"
    vol = (slice(1, 3), slice(92, 184, 1), slice(None))
    a = VCP(21, gridshape, slices=vol, doCycle=False, elevOffset=0)

    print a.radials, len(a.radials), a.U

    for index, b in zip(range(40), a) :
        print b.T, b.currslice
