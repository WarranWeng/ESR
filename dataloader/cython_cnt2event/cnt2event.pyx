# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange, parallel


ctypedef bint TYPE_BOOL
ctypedef unsigned long long TYPE_U_INT64
ctypedef int TYPE_INT32
ctypedef long TYPE_INT64
ctypedef float TYPE_FLOAT
ctypedef double TYPE_DOUBLE


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cnt2event(np.ndarray[float, ndim=4] event_cnt, 
                       int mode):
    """
    event_cnt: numpy.ndarray, [B, 2, H, W], 0 for positive, 1 for negtive
    mode: int, method to assign event timestamp using 0 for linear or 1 for random
    return: numpy.ndarray, batched event cloud: [batch, 2*H*W*max_num_event, 4] [x, y, t, p]
    """
    np.random.seed(123)
    assert event_cnt.shape[1] == 2, 'Wrong event count data!'

    cdef int batch = event_cnt.shape[0]
    cdef int H = event_cnt.shape[2]
    cdef int W = event_cnt.shape[3]
    cdef np.ndarray[float, ndim=4] event_cnt_round = event_cnt.round()
    cdef int maxlen = 0
    cdef np.ndarray[float, ndim=3] batched_event_cloud = np.zeros([batch, 1, 4], dtype=np.float32)

    cdef int batch_idx
    cdef int ecoor_idx
    cdef int ecoor_num
    cdef int lens
    cdef int value
    cdef int num_event
    cdef int max_num_event = event_cnt.max()
    cdef float t0
    cdef float t1
    cdef np.ndarray[float, ndim=3] entry
    cdef np.ndarray[float, ndim=2] entry_tmp
    cdef np.ndarray[float, ndim=2] elist_np
    cdef np.ndarray[long, ndim=2] ecoors
    cdef np.ndarray[long, ndim=1] ecoor
    cdef np.ndarray[float, ndim=2] el
    cdef np.ndarray[float, ndim=2] positive_cnt
    cdef np.ndarray[float, ndim=2] negtive_cnt
    # cdef np.ndarray[float, ndim=4] elist = np.zeros([2, H*W, max_num_event, 4], dtype=np.float32)
    # cdef np.ndarray[float, ndim=3] event_cloud = np.zeros([batch, 2*H*W*max_num_event, 4], dtype=np.float32)

    event_cloud = []
    if event_cnt_round.sum() != 0:
        for batch_idx in range(batch):
            entry = event_cnt_round[batch_idx]
            if entry.sum() != 0:
                elist = []
                positive_cnt = entry[0]
                negtive_cnt = entry[1]

                # process positive events
                ecoors = np.stack(np.nonzero(positive_cnt)).transpose(1, 0) # N x 2, [y, x]
                ecoor_num = ecoors.shape[0]
                for ecoor_idx in range(ecoor_num):
                    ecoor = ecoors[ecoor_idx]
                    value = int(positive_cnt[ecoor[0], ecoor[1]])
                    num_event = value
                    el = np.zeros([num_event, 4], dtype=np.float32) # N x 4, [x, y, t, p]
                    el[:, 0] = np.full([num_event], ecoor[1], dtype=np.float32)
                    el[:, 1] = np.full([num_event], ecoor[0], dtype=np.float32)
                    el[:, 2] = np.linspace(0, 1, num_event) if mode == 0 else np.random.random([num_event])
                    el[:, 3] = np.full([num_event], 1, dtype=np.float32)
                    elist.append(el)
                    # elist[0, ecoor_idx, :num_event] = el

                # process negtive events
                ecoors = np.stack(np.nonzero(negtive_cnt)).transpose(1, 0) # N x 2, [y, x]
                ecoor_num = ecoors.shape[0]
                for ecoor_idx in range(ecoor_num):
                    ecoor = ecoors[ecoor_idx]
                    value = int(negtive_cnt[ecoor[0], ecoor[1]])
                    num_event = value
                    el = np.zeros([num_event, 4], dtype=np.float32) # N x 4, [x, y, t, p]
                    el[:, 0] = np.full([num_event], ecoor[1], dtype=np.float32)
                    el[:, 1] = np.full([num_event], ecoor[0], dtype=np.float32)
                    el[:, 2] = np.linspace(0, 1, num_event) if mode == 0 else np.random.random([num_event])
                    el[:, 3] = np.full([num_event], -1, dtype=np.float32)
                    elist.append(el)
                    # elist[1, ecoor_idx, :num_event] = el

                # sort events by time
                elist_np = np.concatenate(elist)
                # elist_np = elist.reshape(-1, 4)
                elist = sorted(elist_np, key=lambda x: x[2])
                elist_np = np.stack(elist)

            else:
                elist_np = np.zeros([1, 4], dtype=np.float32)
            # elist_np = elist.reshape(-1, 4)

            event_cloud.append(elist_np)
            # event_cloud[batch_idx] = elist.reshape(-1, 4)

        for entry_tmp in event_cloud:
            maxlen = entry_tmp.shape[0] if entry_tmp.shape[0] > maxlen else maxlen

        batched_event_cloud = np.zeros([batch, maxlen, 4], dtype=np.float32)

        for batch_idx in range(batch):
            lens = event_cloud[batch_idx].shape[0]
            batched_event_cloud[batch_idx, :lens, :] = event_cloud[batch_idx]

    return batched_event_cloud
    # return event_cloud


