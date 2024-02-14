import torch
from typing import Union

def to_device(d: dict, device: Union[str, torch.device]):
    '''
        Recursively moves all tensors in d to the device.
    '''
    for k, v in d.items():
        if isinstance(v, dict):
            to_device(v, device)
        elif isinstance(v, torch.Tensor):
            d[k] = v.to(device)

    return d