'''
Utility base classes for dataclasses.
'''
import torch

class DictDataClass:
    '''
        Class used for backward compatibility with code that expected dictionary outputs.
    '''
    def __getitem__(self, key):
        return getattr(self, key)

class DeviceShiftable:
    def to(self, device, detach: bool = False):
        '''
            Detaches and shifts all tensors to the specified device
        '''
        for field in self.__dataclass_fields__:
            attr = getattr(self, field)

            if isinstance(attr, torch.Tensor):
                if detach:
                    attr = attr.detach()

                setattr(self, field, attr.to(device))

        return self

    def cpu(self, detach: bool = True):
        '''
            Moves all tensors to the CPU.
        '''
        return self.to('cpu', detach=detach)

    def cuda(self):
        '''
            Moves all tensors to the GPU.
        '''
        return self.to('cuda')