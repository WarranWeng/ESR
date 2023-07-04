import time
import torch
from . import event_redistribute
import numpy as np
from multiprocessing import Pool


# def python_event_redistribute(event_stack, mode='linear'):
#     """
#     event_stack: [B, P, C, Y, X]
#     mode: method to assign event timestamp using linear or random
#     return batched event cloud: [B, max_num_event, 4] [x, y, t, p]
#     """
#     np.random.seed(123)
#     batch, num_bins = event_stack.size()[:2]

#     event_stack = event_stack.round()
#     event_cloud = []
#     maxlen = 0
#     batched_event_cloud = torch.zeros([batch, 1, 4])

#     if event_stack.sum() != 0:
#         for entry in event_stack:
#             if entry.sum() != 0:
#                 elist = []
#                 ecoors = torch.nonzero(entry) # N x 4, [p, c, y, x] 
#                 for ecoor in ecoors:
#                     value = entry[ecoor[0], ecoor[1], ecoor[2], ecoor[3]]
#                     num_event = int(torch.abs(value).item())
#                     el = torch.zeros([num_event, 4]) # N x 4, [x, y, t, p]
#                     el[:, 0] = torch.full([num_event], ecoor[3])
#                     el[:, 1] = torch.full([num_event], ecoor[2])
#                     t0 = (ecoor[1]/num_bins + 1/(100*num_bins)).item()
#                     t1 = ((ecoor[1]+1)/num_bins).item()
#                     el[:, 2] = torch.linspace(t0, t1, num_event) if mode == 'linear' else torch.from_numpy(np.random.random([num_event]) * (t1-t0) + t0)#torch.rand([num_event]) * (t1-t0) + t0
#                     el[:, 3] = torch.full([num_event], 1 if value > 0 else -1)
#                     elist.append(el)
#                 elist = torch.cat(elist, dim=0)
#                 elist = sorted(elist, key=lambda x: x[2])
#                 elist = torch.stack(elist, dim=0)
#             else:
#                 elist = torch.zeros([1, 4])

#             event_cloud.append(elist)

#         for entry in event_cloud:
#             maxlen = entry.size(0) if entry.size(0) > maxlen else maxlen

#         batched_event_cloud = torch.zeros((batch, maxlen, 4))

#         for batch_idx in range(batch):
#             lens = event_cloud[batch_idx].size(0)
#             batched_event_cloud[batch_idx, :lens, :] = event_cloud[batch_idx]

#     return batched_event_cloud


def python_event_redistribute(event_stack, mode='linear'):
    """
    event_stack: torch.tensor, [B, P, C, Y, X], P refers to polarities(i.e. 2)
    mode: int, method to assign event timestamp using linear or random
    return: torch.tensor, batched event cloud: [B, max_num_event, 4] [x, y, t, p]
    """
    batch = event_stack.size()[0]
    num_bins = event_stack.size()[2]

    event_stack = event_stack.round()
    event_cloud = []
    maxlen = 0
    batched_event_cloud = torch.zeros([batch, 1, 4])

    if event_stack.sum() != 0:
        for entry in event_stack:
            if entry.sum() != 0:
                elist = []
                ecoors = torch.nonzero(entry) # N x 4, [p, c, y, x] 
                for ecoor in ecoors:
                    value = entry[ecoor[0], ecoor[1], ecoor[2], ecoor[3]]
                    num_event = int(torch.abs(value).item())
                    el = torch.zeros([num_event, 4]) # [x, y, t, p]
                    el[:, 0] = torch.full([num_event], ecoor[3])
                    el[:, 1] = torch.full([num_event], ecoor[2])
                    t0 = ecoor[1]/num_bins + 1/(100*num_bins)
                    t1 = (ecoor[1]+1)/num_bins
                    el[:, 2] = torch.linspace(t0, t1, num_event) if mode == 'linear' else torch.rand([num_event]) * (t1-t0) + t0
                    el[:, 3] = torch.full([num_event], 1 if value > 0 else -1)
                    elist.append(el)
                elist = torch.cat(elist, dim=0)
                elist = sorted(elist, key=lambda x: x[2])
                elist = torch.stack(elist, dim=0)
            else:
                elist = torch.zeros([1, 4])

            event_cloud.append(elist)

        for entry in event_cloud:
            maxlen = entry.size(0) if entry.size(0) > maxlen else maxlen

        batched_event_cloud = torch.zeros((len(event_cloud), maxlen, 4))

        for batch_idx in range(len(event_cloud)):
            lens = event_cloud[batch_idx].size(0)
            batched_event_cloud[batch_idx, :lens, :] = event_cloud[batch_idx]

    return batched_event_cloud


def func_wrapper(args):
        return event_redistribute.event_redistribute(*args)


def multiprocess_cython(event_stack, mode):
    """
    params: event_stack: np.ndarray, B x P x C x H x W
    params: mode: int, 0 for 'linear' and 1 for 'random'
    return: event_cloud: np.ndarray, B x N x 4, [x, y, t, p]
    """
    batch = event_stack.shape[0]
    maxlen = 0

    with Pool() as p:
        event_cloud = p.map(func_wrapper, 
                        [[i[np.newaxis, ...], mode] for i in event_stack])

    for entry_tmp in event_cloud:
        maxlen = entry_tmp.shape[1] if entry_tmp.shape[1] > maxlen else maxlen

    batched_event_cloud = np.zeros((batch, maxlen, 4), dtype=np.float32)

    for batch_idx in range(batch):
        lens = event_cloud[batch_idx].shape[1]
        batched_event_cloud[batch_idx, :lens, :] = event_cloud[batch_idx]

    return batched_event_cloud


if __name__ == '__main__':
    event_stack_torch = torch.rand([10, 2, 3, 20, 20]).cuda() * 2
    mode = 0

    t0 = time.time()
    cython_ec = event_redistribute.event_redistribute(event_stack_torch.cpu().numpy().astype(np.float32), mode)
    t1 = time.time()
    cython_time = t1 - t0
    print(f'cython: {t1-t0}')
    print(cython_ec)

    t0 = time.time()
    mp_cython_ec = multiprocess_cython(event_stack_torch.cpu().numpy().astype(np.float32), mode)
    t1 = time.time()
    mp_cython_time = t1 - t0
    print(f'mp_cython: {t1-t0}')
    print(mp_cython_ec)

    t0 = time.time()
    python_ec = python_event_redistribute(event_stack_torch, mode='linear' if mode == 0 else 'random')
    t1 = time.time()
    python_time = t1 - t0
    print(f'python: {t1-t0}')
    print(python_ec)

    print(f'python / cython: {python_time / cython_time}')
    print(f'python / mp_cython: {python_time / mp_cython_time}')
    diff0 = (python_ec.numpy()-cython_ec).sum()
    diff1 = (python_ec.numpy()-mp_cython_ec).sum()
    print(f'differences between cython and python: {diff0}')
    print(f'differences between mp_cython and python: {diff1}')

