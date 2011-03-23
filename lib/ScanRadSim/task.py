from itertools import cycle
import datetime
import numpy as np

from NDIter import SliceIter

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
    def __init__(self, updatePeriod, radials, dwellTime, prt=None, doCycle=False) :
        """
        updatePeriod must be a timedelta from the datetime module.
            It represents the period at which the radials are iterated over

        dwellTime and prt are also timedelta objects
            dwellTime represents how much time it should take to process
            each radial from a call to radials.next()

        radials is any iterator that returns an object that
            can be used to access a part of a numpy array upon
            a call to next().

        doCycle will indicate whether or not to cycle through
        the radials.  Default is False.
        """
        self.U = updatePeriod
        self.is_running = False
        self.radials = cycle(radials) if doCycle else radials
        self.dwellTime = dwellTime
        self.T = None

        if prt is None :
            # For now, assume 10 samples per dwell
            prt = dwellTime / 10

        self.prt = prt

        self.currslice = None

    def __iter__(self) :
        return self

    def _slicesize(self) :
        return int(np.prod([len(range(aSlice.start, aSlice.stop, aSlice.step)) for
                            aSlice in self.currslice[:-1]]))

    def next(self) :
        # TODO: Assume a 10% duty cycle for now...
        self.currslice = self.radials.next()
        #print "In task.next():", self.currslice
        self.T = self.dwellTime * self._slicesize()
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

        ScanJob.__init__(self, updatePeriod, radials, dwellTime, prt, doCycle)


class Surveillance(ScanJob) :
    def __init__(self, dwellTime, gridshape, slices=None, prt=None) :
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

        gridshape = [len(range(*aSlice.indices(shape))) for
                     aSlice, shape in zip(slices, gridshape)]

        # Get the slice-chunking iterator for this task.
        #print "start:", [0] * len(gridshape), " stop:", gridshape
        iterChunk = SliceIter([0] * len(gridshape), gridshape,
                              [1, 5, gridshape[-1]],
                              (1, 0, 2))

        radialCnt = int(np.prod(gridshape[:-1]))
        dwellTime = datetime.timedelta(microseconds=dwellTime)# * radialCnt / len(iterChunk))
        updatePeriod = dwellTime * radialCnt#len(iterChunk)
        #print "Dwell:", dwellTime, "  Radials:", radialCnt, "  ChunkCnt:", len(iterChunk)

        ScanJob.__init__(self, updatePeriod, iterChunk,
                               dwellTime, prt, doCycle=True)

if __name__ == '__main__' :
    a = Surveillance(6400, (9, 366, 1000), slices=(slice(None), slice(0, 92, 1), slice(None)))

    print a.radials, len(a.radials)

    
    for index in range(20) :
        #print a.radials._chunkIndices, a.radials._chunkCnts
        b = a.next()
 
