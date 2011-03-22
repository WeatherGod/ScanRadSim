from itertools import cycle
import datetime
import numpy as np

from NDIter import SliceIter

class ScanOperation(object) :
    def __init__(self, radSlice, tx_time, rx_time, wait_time=0) :
        """
        Times for the three parts of any scan operation in microseconds (integers only!).
        tx == transmit
        rx == receive

        A Scan Operation can not be pre-empted during the transmit and receive modes.
        """
        self.tx_time = tx_time
        self.rx_time = rx_time
        self.wait_time = wait_time
        self.radSlice = radSlice


class ScanJob(object) :
    def __init__(self, updatePeriod, radials, dwellTime, prt=None, doCycle=False) :
        """
        updatePeriod must be a timedelta from the datetime module.
            It represents the period at which the radials are iterated over

        dwellTime and prt are also timedelta objects

        radials is any iterator that returns an object that
            can be used to access a part of a numpy array upon
            a call to next().

        doCycle will indicate whether or not to cycle through
        the radials.  Default is False.
        """
        self.U = updatePeriod
        self.is_running = False
        self.radials = cycle(radials) if doCycle else radials
        self.dwell_time = dwellTime

        if prt is None :
            # For now, assume 10 samples per dwell
            prt = dwellTime / 10

        self.prt = prt

        self.currslice = None

    def __iter__(self) :
        return self

    def next(self) :
        self.currslice = self.radials.next()
        return self.currslice

class StaticJob(ScanJob) :
    def __init__(self, updatePeriod, radials, dwellTime, prt=None, doCycle=True) :
        """
        updatePeriod must be a timedelta from datetime module and represents the
            period of time between calls to next() from the radials iterator.
        dwellTime and prt are also timedelta objects

        radials will be any iterator that returns an item that can be used to
            access a part or sector of a numpy array upon a call to next()
        """
        timeToComplete = len(radials) * dwellTime
        if updatePeriod < timeToComplete :
            updatePeriod = timeToComplete

        self.T = timeToComplete
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

        radialCnt = int(np.prod(gridshape[:-1])) / 6
        updatePeriod = datetime.timedelta(microseconds=dwellTime * radialCnt * 6)
        #print "Surveillance Grid:", gridshape, radialCnt, dwellTime, updatePeriod
        
        ## How many radials can we process within a time fragment?
        #chunkLen = int(((timeFragment.seconds * 1e6) + timeFragment.microseconds) // dwellTime)

        #print "ChunkLen:", chunkLen, "   GridShape:", gridshape


        # Get the slice-chunking iterator for this task.
        iterChunk = SliceIter([0] * len(gridshape), gridshape,
                              [1, gridshape[1] / 6, gridshape[-1]],
                              (1, 0, 2))

        self.T = updatePeriod / 6
        ScanJob.__init__(self, updatePeriod, iterChunk,
                               dwellTime, prt, doCycle=True)
 
