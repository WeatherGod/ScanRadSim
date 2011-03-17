from itertools import cycle
import datetime
import numpy as np

class Task(object) :
    def __init__(self, updatePeriod, timeFragment, radials, dwellTime=None, prt=None) :
        """
        updatePeriod, timeFragment must be timedeltas from datetime module.
        dwellTime and prt must be ints in units of microseconds.

        radials will be any iterator that returns an item that can be used to
        access a part or sector of a numpy array upon a call to next()
        """
        self.U = updatePeriod
        self.T = timeFragment

        # Just keep doing these radials over and over...
        self.radials = cycle(radials)

        if dwellTime is None :
            dwellTime = timeFragment * len(radials)
        if prt is None :
            # For now, just assume ten samples per dwell
            prt = timeFragment / (10 * len(radials))

        self.dwellTime = dwellTime
        self.prt = prt

    def __iter__(self) :
        return self

    def next(self) :
        return self.radials.next()


class Surveillance(Task) :
    def __init__(self, timeFragment, dwellTime, gridshape, prt=None) :
        """
        timeFragment must be a timedelta object from the datetime module.
        dwellTime and prt must be an int in units of microseconds.
        gridshape must be a tuple of ints representing the shape of the
            numpy array representing the radar grid.
        """
        scanVol = [slice(0, shape, None) for shape in gridshape]
        radialCnt = np.prod(gridshape[:-1])
        updatePeriod = datetime.timedelta(microseconds=dwellTime * radialCnt)
        
        # How many radials can we process within a time fragment?
        chunkLen = int(((timeFragment.seconds * 1e6) + timeFragment.microseconds) // dwellTime)

        if prt is None :
            # For now, just assume ten samples per dwell...
            prt = int(timeFragment.seconds * 1e6 + timeFragment.microseconds) // (10 * radialCnt)

        Task.__init__(self, updatePeriod, timeFragment,
                            ChunkIter(scanVol, chunkLen, radialCnt),
                            dwellTime, prt)



class ChunkIter(object) :
    """
    Takes a scanVolume, and wraps it in a manner that
    would make a call to this object's next() function return
    slices that would represent the next chunk of radials.

    In other words, for a given scanVol and chunksize, produce
    an iterator that returns a slice that represents a chunk of
    scanVol on each call to next()
    """
    def __init__(self, scanVol, chunksize) :
        self._radIndex = 0
        self._start = scanVol[0].start
        self._stop = scanVol[0].stop
        self.chunksize = chunksize

    def __iter__(self) :
        return self

    def next(self) :
        start = self._radIndex
        stop = (self._radIndex + self.chunksize) % (self._stop - self._start)
        self._radIndex = stop
        return (slice(start + self._start, stop + self._start, None), slice(None, None, None))


