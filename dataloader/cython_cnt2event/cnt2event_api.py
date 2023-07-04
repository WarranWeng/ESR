from . import cnt2event
import numpy as np
import torch


def cnt2eventAPI_tmp(event_cnt, mode=0):
    """
    event cnt: torch.tensor, Bx2xHxW
    mode: int, method to assign event timestamp using 0 for linear or 1 for random
    return: torch.tensor, batched event cloud: [B, max_num_event, 4] [x, y, t, p]
    """
    events = cnt2event.cnt2event(event_cnt, mode) # [batch, 2*H*W*max_num_event, 4] [x, y, t, p]

    final_events = []
    for batch_events in events:
        mask = batch_events[:, -1] != 0
        valid_events = batch_events[mask]
        elist = sorted(valid_events, key=lambda x: x[2])
        valid_events = np.stack(elist)
        final_events.append(valid_events)

    return np.stack(final_events)


def cnt2eventAPI(event_cnt, mode=0):
    """
    event cnt: torch.tensor, Bx2xHxW
    mode: int, method to assign event timestamp using 0 for linear or 1 for random
    return: torch.tensor, batched event cloud: [B, max_num_event, 4] [x, y, t, p]
    """
    cnt_np = event_cnt.detach().cpu().numpy().astype(np.float32)
    events = cnt2event.cnt2event(cnt_np, mode) # [batch, max_num_event, 4] [x, y, t, p]
    events_torch = torch.from_numpy(events)

    return events_torch
