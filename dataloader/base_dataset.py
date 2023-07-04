import torch
from torch.utils.data import Dataset
from abc import abstractmethod
import numpy as np
import random
# from dataloader.binary_search import binary_search


class BaseDataset(Dataset):
    """
    Base class for dataset
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def event_formatting(events):
        
        xs = torch.from_numpy(events[0].astype(np.float32))
        ys = torch.from_numpy(events[1].astype(np.float32))
        ts = torch.from_numpy(events[2].astype(np.float32))
        ps = torch.from_numpy(events[3].astype(np.float32))
        ts = (ts - ts[0]) / (ts[-1] - ts[0] + 1e-6)
        return torch.stack([xs, ys, ts, ps])

    @staticmethod
    def frame_formatting(frame):

        return torch.from_numpy(frame.astype(np.uint8)).float().unsqueeze(0) / 255

    @staticmethod
    def binary_search_h5_dset(dset, x, l=None, r=None, side='left'):
        # interpolation search****************************************************
        # l = 0 if l is None else l
        # r = len(dset)-1 if r is None else r

        # while l <= r:
        #     interval = int((r - l) * (x - dset[l]) / (dset[r] - dset[l] + 1e-6))
        #     mid = l + interval

        #     midval = dset[mid]
        #     if midval == x:
        #         return mid
        #     elif midval < x:
        #         l = mid + 1
        #         if dset[l] >= x:
        #             return l
        #     else:
        #         r = mid - 1
        #         if dset[r] <= x:
        #             return r

        # if side == 'left':
        #     return l

        # return r


        # binary search ********************************************
        l = 0 if l is None else l
        r = len(dset)-1 if r is None else r

        while l <= r:
            mid = l + (r - l)//2
            midval = dset[mid]
            if midval == x:
                return mid
            elif midval < x:
                l = mid + 1
            else:
                r = mid - 1

        if side == 'left':
            return l


        # cython ***************************************************
        # if not isinstance(dset, np.ndarray):
        #     dset = np.array(dset)

        # return binary_search.binary_search_func(dset, x)


        # numpy search *****************************************
        # return np.searchsorted(dset, x)