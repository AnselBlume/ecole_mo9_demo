'''
Utility base classes for dataclasses.
'''
import torch
from typing import Callable

class DictDataClass:
    '''
        Class used for backward compatibility with code that expected dictionary outputs.
    '''
    def __getitem__(self, key):
        return getattr(self, key)

class DeviceShiftable:
    def apply_to_tensors(self, func: Callable[[torch.Tensor], torch.Tensor]):
        '''
            Applies a function to all tensors in the dataclass.
        '''
        for field in self.__dataclass_fields__:
            attr = getattr(self, field)

            if isinstance(attr, torch.Tensor):
                setattr(self, field, func(attr))

        return self

    def to(self, device):
        '''
            Shifts all tensors to the specified device
        '''
        return self.apply_to_tensors(lambda x: x.to(device))

    def cpu(self):
        '''
            Moves all tensors to the CPU.
        '''
        return self.to('cpu')

    def cuda(self):
        '''
            Moves all tensors to the GPU.
        '''
        return self.to('cuda')

    def detach(self):
        '''
            Detaches all tensors.
        '''
        return self.apply_to_tensors(lambda x: x.detach())
