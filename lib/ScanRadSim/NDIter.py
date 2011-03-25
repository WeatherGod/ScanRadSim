from itertools import cycle
import numpy as np


class BaseNDIter(object) :
    def __init__(self, chunkIters, chunkCnts, cycleList=None) :
        if cycleList is None :
            cycleList = range(len(chunkIters))

        # So that I know which axes change more than others.
        self._cycleList = cycleList

        # So that I know how many chunks are in each axes.
        self._chunkCnts = chunkCnts

        # ChunkIndices for keeping track of when an iterator cycles
        # Initialize to self._chunkCnts so that the first call to
        # .next() will force an initialization of self.slices
        #
        # Also, make sure you make a copy!
        self._chunkIndices = self._chunkCnts[:]

        # The slice iterators for each axis (besides the range-gate one)
        self._chunkIters = chunkIters

        # This member will contain the current slices.
        self.slices = [None] * len(chunkIters)
        self._started = False

    def __len__(self) :
        return np.prod(self._chunkCnts)

    def __iter__(self) :
        return self

    def next(self) :
        for axisIndex in self._cycleList :
            self._chunkIndices[axisIndex] += 1
            self.slices[axisIndex] = self._chunkIters[axisIndex].next()

            if self._chunkIndices[axisIndex] >= self._chunkCnts[axisIndex] :

                # This axis needs cycling.
                if (axisIndex != self._cycleList[-1] or
                    not self._started) :
                    # If we are not trying to wrap the last axis, then carry on!
                    self._chunkIndices[axisIndex] = 0
                else :
                    # This is the last axis, so let's stop the iteration.
                    raise StopIteration
            else :
                # This axis didn't need cycling, so we don't need to worry about
                # the rest of the axes this time around.
                break

        self._started = True
        #print "In next():", self.slices
        return self.slices[:]

class SliceIter(BaseNDIter) :
    """
    An iterator that generate slices to access chunks of
    arrays of the same shape.
    """
    def __init__(self, starts, stops, steps, cycleList=None) :
        if cycleList is None :
            cycleList = range(len(starts))

        div_points = [range(start, stop, step) + [stop] for
                      start, stop, step in zip(starts, stops, steps)]


        # So that I know how many chunks are in each axes.
        chunkCnts = [len(div) - 1 for div in div_points]

        chunkIters = [cycle([slice(start, stop, np.sign(step)) for start, stop
                             in zip(divs[:-1], divs[1:])]) for
                      divs, step in zip(div_points, steps)]

        BaseNDIter.__init__(self, chunkIters, chunkCnts, cycleList)




class SplitIter(BaseNDIter) :
    """
    An iterator that generates slices to access chunks of
    arrays of the same shape.
    """

    def __init__(self, gridshape, indices_or_sections, axis=0, slices=None) :
        """
        Create a SplitIter iterator that would generate
        slices to arrays of shape `gridshape`. The slices
        would each access array sections of equal or near-equal sizes.

        Parameters
        ----------
        gridshape : array-like
            Shape of the array(s) being split.

        indices_or_sections : int or 1-D array
            if `indices_or_sections` is an integer, N, the array will
            be divided into N sections along `axis`. N does *not*
            have to equally divide the axis.

            If `indices_or_sections` is a 1-D array of sorted integers,
            the entries indicate where along `axis` the array is split.
            For example, ``[2, 3]`` would, for ``axis=0``, result in
            splitting the first axis from 0 to 2, then 2 to 3,
            then 3 to end.

            If an index exceeds the dimension of the array along `axis`,
            an empty slice is generated for that portion.

        axis : int, optional
            The axis along which to split, default is 0

        slices : array-like of slice objects, optional
            list of slices representing where in the given `gridshape`
            the chunks should iterate over. If it is None, then assume
            that the entire region being accessed.

        Chunking will occur along the specified axis, while
        all other axes will be iterated over, one-by-one.
        """
        if slices is None :
            slices = [slice(None) for size in gridshape]

        indices = [aSlice.indices(shape) for aSlice, shape in zip(slices, gridshape)]

        #print slices, axis, gridshape
        Ntotal = len(range(*indices[axis]))

        try:
            Nsections = len(indices_or_sections) + 1
            div_points = [slices[axis].start] + list(indices_or_sections) +\
                         [slices[axis].stop]
        except TypeError:  # indices_or_sections is a scalar, not an array
            Nsections = int(indices_or_sections)
            if Nsections <= 0 :
                raise ValueError("Must be at least one section")
            Neach_section, extras = divmod(Ntotal, Nsections)

            section_sizes = [0] + \
                            ([Neach_section+1] * extras) + \
                            ([Neach_section] * (Nsections-extras))

            div_points = np.cumsum(section_sizes)
            if np.sign(indices[axis][2]) >= 0 :
                div_points += indices[axis][0]
            else :
                div_points += indices[axis][1]
                div_points = div_points[::-1]

        otherAxes = range(len(gridshape))
        otherAxes.remove(axis)

        cycleList = [axis] + otherAxes

        chunkCnts = [len(range(*indx)) for indx in indices]
        chunkCnts[axis] = Nsections

        chunkIters = [None] * len(gridshape)

        for index in range(len(gridshape)) :    
            if index in otherAxes :
                tmp_divPts = range(*indices[index])
                # add another point so that we can iterate all the way through.
                tmp_divPts += [indices[index][1]]
            else :
                tmp_divPts = div_points

            chunkIters[index] = cycle([slice(start, stop, indices[index][2]) for
                                       start, stop in zip(tmp_divPts[:-1], tmp_divPts[1:])])

        BaseNDIter.__init__(self, chunkIters, chunkCnts, cycleList)




       
class ChunkIter(SplitIter) :
    """
    Similar to SplitIter, but instead allows you to specify how large the
    chunks should be, and it will determine the best fit.

    Also, for the purposes of this project (but likely factored out later),
    we assume that the last dimension is never considered for chunk fitting
    and is sliced in its entirety.
    
    If no fit can be made, then throw an error.
    """
    def __init__(self, gridshape, chunksize, slices=None) :
        if len(gridshape) < 2 :
            raise ValueError("The slices being chunked must be for at least a 2-D array.")

        if chunksize <= 0 :
            raise ValueError("chunksize must be greater than zero")

        if slices is None :
            slices = [slice(None) for size in gridshape]

        views = [aSlice.indices(size) for aSlice, size in zip(slices, gridshape)]

        # Rebuild the slices to make sure every part is defined.
        #slices = [slice(*aView) for aView in views]

        # Find out how many chunksize sections can fit in each axis,
        # and how many remaining elements in the remaining mis-fit section.
        Nfitsects, extras = zip(*[divmod(len(range(*aView)), chunksize) for
                                  aView in views[:-1]])

        Nfitsects = np.array(Nfitsects)
        extras = np.array(extras)
        
        # The divmod is zero if chunksize is greater than size.
        if np.all(Nfitsects == 0) :
            raise ValueError("chunksize must be smaller than or equal to at least one of the dimensions")

        if 0 in extras :
            # One of the axes fitted perfectly!
            axis = extras.tolist().index(0)
            chunkCnt = Nfitsects[axis]
        else :
            # Since none fitted perfectly, we want the most
            # efficient fit, which means that the mis-fit
            # section should still be as close as possible
            # to the expected chunksize.
            # We find the axis that has the best packing efficiency
            packing = ((extras + (chunksize * Nfitsects)) /
                       (chunksize * (Nfitsects + 1.0)))
            axis = np.argmax(packing)
            chunkCnt = Nfitsects[axis] + 1

        SplitIter.__init__(self, gridshape[:-1], chunkCnt, axis,
                           [aSlice for aSlice in slices[:-1]])

        # Add a final element onto this tuple representing the
        # desired slice for the final dimension.
        self.slices += [slices[-1]]




if __name__ == '__main__' :

    #--------------------------------------------------------
    a = ChunkIter((40, 5, 1000), 20, (slice(39, 0, -1), slice(None, 0, -1), slice(0, 1000, None)))

    print "Cycle List:", a._cycleList, "  ChunkCnts:", a._chunkCnts, len(a)
    print a.slices, "  |||  ", a._chunkIndices

    for index, theSlice in zip(range(25), a) :
        print theSlice
    #-------------------------------------------------------
    a = SplitIter((9, 366,), 6, axis=1)

    print "\nCycle List:", a._cycleList, "  ChunkCnts:", a._chunkCnts, len(a)
    print a.slices, "  |||  ", a._chunkIndices

    for index, theSlice in zip(xrange(25), a) :
        print theSlice

    #-------------------------------------------------------

    a = SliceIter((0, 91, 0), (9, 2, 1000), (1, -5, 1000), (1, 0, 2))
    print "\nCycle List:", a._cycleList, "  ChunkCnts:", a._chunkCnts, len(a)
    print a.slices, "  |||  ", a._chunkIndices

    for index in range(20) :
        print a.next()
