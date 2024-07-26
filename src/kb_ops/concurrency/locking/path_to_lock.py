from __future__ import annotations
from .lockable import Lockable
from abc import ABC

class PathToLockMapping(ABC):
    def __getitem__(self, path: str) -> Lockable:
        '''
            Get a lock for the given path. If the lock does not exist, it is created.
        '''
        return self.get_lock(path)

    def get_lock(self, path: str) -> Lockable:
        '''
            Get a lock for the given path.

            Args:
                path (str): Path to the file
        '''
        raise NotImplementedError('get_lock must be implemented by subclasses')

    def __contains__(self, path: str) -> bool:
        '''
            Check if a lock exists for the given path.

            Args:
                path (str): Path to the file
        '''
        raise NotImplementedError('__contains__ must be implemented by subclasses')