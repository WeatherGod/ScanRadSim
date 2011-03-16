import numpy as np

def interp_radar(raddata_t0, raddata_t1, times, motion_vects=None, XYZ=None) :
    """
    raddata_t0 and raddata_t1 are parallel arrays (2-D or 3-D)

    times is a list of times for when we want resulting interpolated
    data for.  times should be normalized
    against the times for raddata_t0 (0.0) and raddata_t1 (1.0).

    motion_vects is an array of the same shape as raddata_t0
    and raddata_t1 that indicates the motion vector for each point
    in raddata_t0 and raddata_t1.  DON'T USE YET!

    XYZ is a tuple of arrays (each the same shape as raddata_t0),
    providing the cartesian coordinate values in each axis.
    The number of elements in the tuple matches the number of
    dimensions in the input data.  DON'T USE YET!
    """
    """
    if motion_vects is None :
        motion_vects = np.zeros_like(raddata_t0)

    if XYZ is None :
        XYZ = np.ix_(*[np.linspace(0, 1, length) for
                       length in raddata_t0.shape])
    """
    # Ignore motion data for now.  Assume motion is zero.
    # If motion was non-zero, I would calculate the new
    # location of each point in XYZ and motion_vect, and
    # use that information to "realign" raddata_t0 to the
    # grid, taking care to properly handle the edges and
    # maybe even some convergence/divergence issues?
    #
    # Probably too complicated for now...
    #

    slope = raddata_t1 - raddata_t0
    return [(aTime * slope + raddata_t0) for aTime in times]

