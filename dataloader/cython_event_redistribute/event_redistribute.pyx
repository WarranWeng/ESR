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
def event_redistribute_PolarityStack(np.ndarray[float, ndim=5] event_stack, 
                       int mode):
    """
    event_stack: numpy.ndarray, [B, P, C, Y, X]
    mode: int, method to assign event timestamp using 0 for linear or 1 for random
    return: numpy.ndarray, batched event cloud: [B, max_num_event, 4] [x, y, t, p]
    """
    np.random.seed(123)
    cdef int batch = event_stack.shape[0]
    cdef int num_bins = event_stack.shape[2]
    cdef np.ndarray[float, ndim=5] event_stack_round = event_stack.round()
    cdef int maxlen = 0
    cdef np.ndarray[float, ndim=3] batched_event_cloud = np.zeros([batch, 1, 4], dtype=np.float32)

    cdef int batch_idx
    cdef int ecoor_idx
    cdef int ecoor_num
    cdef int lens
    cdef int value
    cdef int num_event
    cdef float t0
    cdef float t1
    cdef np.ndarray[float, ndim=4] entry
    cdef np.ndarray[float, ndim=2] entry_tmp
    cdef np.ndarray[float, ndim=2] elist_np
    cdef np.ndarray[long, ndim=2] ecoors
    cdef np.ndarray[long, ndim=1] ecoor
    cdef np.ndarray[float, ndim=2] el

    event_cloud = []
    if event_stack_round.sum() != 0:
        for batch_idx in range(batch):
            entry = event_stack_round[batch_idx]
            if entry.sum() != 0:
                elist = []
                ecoors = np.stack(np.nonzero(entry)).transpose(1, 0) # N x 4, [p, c, y, x] 
                ecoor_num = ecoors.shape[0]
                for ecoor_idx in range(ecoor_num):
                    ecoor = ecoors[ecoor_idx]
                    value = int(entry[ecoor[0], ecoor[1], ecoor[2], ecoor[3]])
                    num_event = int(np.abs(value))
                    el = np.zeros([num_event, 4], dtype=np.float32) # N x 4, [x, y, t, p]
                    el[:, 0] = np.full([num_event], ecoor[3], dtype=np.float32)
                    el[:, 1] = np.full([num_event], ecoor[2], dtype=np.float32)
                    t0 = ecoor[1]/num_bins + 1/(100*num_bins)
                    t1 = (ecoor[1]+1)/num_bins
                    el[:, 2] = np.linspace(t0, t1, num_event) if mode == 0 else np.random.random([num_event]) * (t1-t0) + t0
                    el[:, 3] = np.full([num_event], 1 if value > 0 else -1, dtype=np.float32)
                    elist.append(el)
                elist_np = np.concatenate(elist)
                elist = sorted(elist_np, key=lambda x: x[2])
                elist_np = np.stack(elist)
            else:
                elist_np = np.zeros([1, 4], dtype=np.float32)

            event_cloud.append(elist_np)

        for entry_tmp in event_cloud:
            maxlen = entry_tmp.shape[0] if entry_tmp.shape[0] > maxlen else maxlen

        batched_event_cloud = np.zeros([batch, maxlen, 4], dtype=np.float32)

        for batch_idx in range(batch):
            lens = event_cloud[batch_idx].shape[0]
            batched_event_cloud[batch_idx, :lens, :] = event_cloud[batch_idx]

    return batched_event_cloud


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def event_redistribute_NoPolarityStack(np.ndarray[float, ndim=4] event_stack, 
                       int mode):
    """
    event_stack: numpy.ndarray, [B, C, Y, X]
    mode: int, method to assign event timestamp using 0 for linear or 1 for random
    return: numpy.ndarray, batched event cloud: [B, max_num_event, 4] [x, y, t, p]
    """
    np.random.seed(123)
    cdef int batch = event_stack.shape[0]
    cdef int num_bins = event_stack.shape[1]
    cdef np.ndarray[float, ndim=4] event_stack_round = event_stack.round()
    cdef int maxlen = 0
    cdef np.ndarray[float, ndim=3] batched_event_cloud = np.zeros([batch, 1, 4], dtype=np.float32)

    cdef int batch_idx
    cdef int ecoor_idx
    cdef int ecoor_num
    cdef int lens
    cdef int value
    cdef int num_event
    cdef float t0
    cdef float t1
    cdef np.ndarray[float, ndim=3] entry
    cdef np.ndarray[float, ndim=2] entry_tmp
    cdef np.ndarray[float, ndim=2] elist_np
    cdef np.ndarray[long, ndim=2] ecoors
    cdef np.ndarray[long, ndim=1] ecoor
    cdef np.ndarray[float, ndim=2] el

    event_cloud = []
    if event_stack_round.sum() != 0:
        for batch_idx in range(batch):
            entry = event_stack_round[batch_idx]
            if entry.sum() != 0:
                elist = []
                ecoors = np.stack(np.nonzero(entry)).transpose(1, 0) # N x 3, [c, y, x] 
                ecoor_num = ecoors.shape[0]
                for ecoor_idx in range(ecoor_num):
                    ecoor = ecoors[ecoor_idx]
                    value = int(entry[ecoor[0], ecoor[1], ecoor[2]])
                    num_event = int(np.abs(value))
                    el = np.zeros([num_event, 4], dtype=np.float32) # N x 4, [x, y, t, p]
                    el[:, 0] = np.full([num_event], ecoor[2], dtype=np.float32)
                    el[:, 1] = np.full([num_event], ecoor[1], dtype=np.float32)
                    t0 = ecoor[0]/num_bins + 1/(100*num_bins)
                    t1 = (ecoor[0]+1)/num_bins
                    el[:, 2] = np.linspace(t0, t1, num_event) if mode == 0 else np.random.random([num_event]) * (t1-t0) + t0
                    el[:, 3] = np.full([num_event], 1 if value > 0 else -1, dtype=np.float32)
                    elist.append(el)
                elist_np = np.concatenate(elist)
                elist = sorted(elist_np, key=lambda x: x[2])
                elist_np = np.stack(elist)
            else:
                elist_np = np.zeros([1, 4], dtype=np.float32)

            event_cloud.append(elist_np)

        for entry_tmp in event_cloud:
            maxlen = entry_tmp.shape[0] if entry_tmp.shape[0] > maxlen else maxlen

        batched_event_cloud = np.zeros([batch, maxlen, 4], dtype=np.float32)

        for batch_idx in range(batch):
            lens = event_cloud[batch_idx].shape[0]
            batched_event_cloud[batch_idx, :lens, :] = event_cloud[batch_idx]

    return batched_event_cloud