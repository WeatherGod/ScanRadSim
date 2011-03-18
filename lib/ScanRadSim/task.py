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
        radialCnt = int(np.prod(gridshape[:-1]))
        updatePeriod = datetime.timedelta(microseconds=dwellTime * radialCnt)
        
        # How many radials can we process within a time fragment?
        chunkLen = int(((timeFragment.seconds * 1e6) + timeFragment.microseconds) // dwellTime)

        print "ChunkLen:", chunkLen, "   GridShape:", gridshape

        if prt is None :
            # For now, just assume ten samples per dwell...
            prt = int(timeFragment.seconds * 1e6 + timeFragment.microseconds) // (10 * radialCnt)

        Task.__init__(self, updatePeriod, timeFragment,
                            ChunkIter(scanVol, chunkLen),
                            dwellTime, prt)



class ChunkIter(object) :
    """
    Takes a scanVolume, and wraps it in a manner that
    would make a call to this object's next() function return
    slices that would represent the next chunk of radials.

    In other words, for a given scanVol and chunksize, produce
    an iterator that returns a slice that represents a chunk of
    scanVol on each call to next()

    Chunking will occur along the axis that best fits.

    We assume that the last dimension is the range-gate
    dimension and should never be chunked. Any remaining
    dimensions are set to have a chunksize of 1.
    """
    def __init__(self, scanVol, chunksize) :
        if len(scanVol) < 2 :
            raise ValueError("The slices being chunked must be for at least a 2-D array.")

        if chunksize <= 0 :
            raise ValueError("chunksize must be greater than zero")

        # I want all but the last dimension, which should be the range-gate dimension.
        gridshape = [aSlice.stop - aSlice.start for aSlice in scanVol[:-1]]

        # Note that any axis with a length shorter than the requested chunksize will
        # have a resulting value that is capped at the length of that axis.
        Nfitsects, extras = zip(*[divmod(size, chunksize) for size in gridshape])
        
        # The divmod is zero if chunksize is greater than size.
        if np.all(np.array(Nfitsects) == 0) :
            raise ValueError("chunksize must be smaller than or equal to at least one of the dimensions")

        if 0 in extras :
            # One of the axes fitted perfectly!
            axis = extras.index(0)

            section_sizes = [0] + ([chunksize] * Nfitsects[axis])
            chunkCnt = Nfitsects[axis]
        else :
            # Since none fitted perfectly, we want the most
            # efficient fit, which means that the mis-fit
            # section should still be as close as possible
            # to the expected chunksize.
            axis = np.argmax(extras)
            section_sizes = [0, extras[axis]] + \
                            [chunksize] * Nfitsects[axis]
            chunkCnt = Nfitsects[axis] + 1

        div_points = np.array(section_sizes).cumsum()

        otherAxes = range(len(scanVol) - 1)
        otherAxes.remove(axis)

        # So that I know which axes change more than others.
        self._cycleList = [axis] + otherAxes

        # So that I know how many chunks are in each axes.
        self._chunkCnts = gridshape
        self._chunkCnts[axis] = chunkCnt

        # ChunkIndices for keeping track on when an iterator cycles
        # Initialize to self._chunkCnts so that the first call to
        # .next() will force an initialization of self.slices
        #
        # Also, make sure you make a copy!
        self._chunkIndices = self._chunkCnts[:]

        # The slice iterators for each axis (besides the range-gate one)
        self._chunkIters = [None] * (len(scanVol) - 1)

        for index in range(len(scanVol) - 1) :
            if index in otherAxes :
                tmp_divPts = range(gridshape[index] + 1)
            else :
                tmp_divPts = div_points

            self._chunkIters[index] = cycle(slice(start, stop, None) for start, stop
                                            in zip(tmp_divPts[:-1], tmp_divPts[1:]))

        # This member will contain the current slices.
        # Remember that the last axis will always be sliced completely
        self.slices = ([None] * (len(scanVol) - 1)) + [slice(None, None, None)]

    def __iter__(self) :
        return self

    def next(self) :
        for axisIndex in self._cycleList :
            self._chunkIndices[axisIndex] += 1
            self.slices[axisIndex] = self._chunkIters[axisIndex].next()
            if self._chunkIndices[axisIndex] >= self._chunkCnts[axisIndex] :
                self._chunkIndices[axisIndex] = 0
            else :
                break
        return self.slices
