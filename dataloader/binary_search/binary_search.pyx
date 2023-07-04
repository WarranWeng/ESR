# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython


ctypedef bint TYPE_BOOL
ctypedef unsigned long long TYPE_U_INT64
ctypedef int TYPE_INT32
ctypedef long TYPE_INT64
ctypedef float TYPE_FLOAT
ctypedef double TYPE_DOUBLE


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def binary_search_func(np.ndarray[double, ndim=1] dset, double x):
    """
    dset: numpy.ndarray
    x: double
    return: idx: int
    """
    cdef int l = 0
    cdef int r = dset.shape[0] - 1
    cdef int mid
    cdef double midval

    while l <= r:
        mid = l + (r - l) // 2
        midval = dset[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1

    return l








