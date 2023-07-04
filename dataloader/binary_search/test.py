import time
import torch
import numpy as np
import binary_search


def binary_search_h5_dset(dset, x, l=None, r=None, side='left'):
    l = 0 if l is None else l
    r = len(dset)-1 if r is None else r

    while l <= r:
        mid = l + (r - l)//2;
        midval = dset[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1

    if side == 'left':
        return l

    return r


NUM = int(128053)
query = 38.0
dset = np.linspace(1, NUM, num=NUM).astype('float')
x = query
print('***************************')
print(f'query:{query}')

t0 = time.time()
idx = binary_search.binary_search_func(dset, x)
t1 = time.time()
cython_time = t1 - t0
print(f'cython time: {t1-t0}')
print(f'cython idx: {idx}')
print(f'cython result: {dset[idx]}')

print('***************************')

t0 = time.time()
idx = binary_search_h5_dset(dset, x)
t1 = time.time()
python_time = t1 - t0
print(f'python time: {t1-t0}')
print(f'python idx: {idx}')
print(f'python result: {dset[idx]}')

print('***************************')
print(f'python over cython:{python_time/cython_time}')





