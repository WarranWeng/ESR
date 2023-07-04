import time
import torch
import cnt2event
import numpy as np


if __name__ == '__main__':
    event_cnt_torch = torch.rand([2, 2, 2, 2]) * 2
    event_cnt_torch = event_cnt_torch.round()
    mode = 0

    t0 = time.time()
    cython_ec = cnt2event.cnt2event(event_cnt_torch.cpu().numpy().astype(np.float32), mode)
    t1 = time.time()
    cython_time = t1 - t0
    print(f'cython: {t1-t0}')
    print(f'cnt: {event_cnt_torch}')
    print(f'events: {cython_ec}')












