from __future__ import annotations
from abc import ABC
from .lockable import Lockable
from readerwriterlock.rwlock import RWLockFair, RWLockable
from .multiprocessing_lock_adapter import MultiprocessingToThreadingLockAdapter
from filelock import FileLock
from typing import Any
from enum import Enum

class LockType(Enum):
    '''
        Enum to represent the type of lock.
    '''
    READERS_WRITERS_LOCK = 'readers_writers_lock'
    FILE_LOCK = 'file_lock'

def get_lock_generator(lock_type: LockType, **path_to_lock_map_init_kwargs) -> LockGenerator:
    '''
        Get a PathToLockMap instance for the given lock type.
    '''
    if lock_type == LockType.READERS_WRITERS_LOCK:
        return ReadersWritersLockGenerator(**path_to_lock_map_init_kwargs)
    elif lock_type == LockType.FILE_LOCK:
        return FileLockGenerator(**path_to_lock_map_init_kwargs)
    else:
        raise ValueError(f'Invalid lock type: {lock_type}')

class LockGenerator(ABC):
    def __init__(self, **lock_init_kwargs):
        self.lock_init_kwargs = lock_init_kwargs

    def get_lock(self, path: str, is_reader: bool = None, **lock_init_kwargs_override) -> Lockable:
        raise NotImplementedError('get_lock must be implemented by subclasses')

    def _consolidate_lock_init_kwargs(self, lock_init_kwargs_override: dict) -> dict[str, Any]:
        lock_init_kwargs = self.lock_init_kwargs.copy()
        lock_init_kwargs.update(lock_init_kwargs_override)
        return lock_init_kwargs

class FileLockGenerator(LockGenerator):
    def get_lock(self, path: str, is_reader: bool = None, **lock_init_kwargs_override) -> Lockable:
        if is_reader is not None:
            raise ValueError('is_reader must be None for file locks')

        lock_init_kwargs = self._consolidate_lock_init_kwargs(lock_init_kwargs_override)

        return FileLock(path, **lock_init_kwargs)

class ReadersWritersLockGenerator(LockGenerator):
    def __init__(self, rw_lockable_class: RWLockable = RWLockFair, lock_factory=MultiprocessingToThreadingLockAdapter, **lock_init_kwargs):
        lock_init_kwargs['lock_factory'] = lock_factory
        super().__init__(**lock_init_kwargs)

        self.rw_lockable_class = rw_lockable_class
        self.path_to_file_lock: dict[str, RWLockable] = {}

    def get_lock(self, path: str, is_reader: bool = None, **lock_init_kwargs_override) -> Lockable:
        if is_reader is None:
            raise ValueError('is_reader must not be None for rw locks')

        lock_init_kwargs = self._consolidate_lock_init_kwargs(lock_init_kwargs_override)

        if path not in self.path_to_file_lock:
            self.path_to_file_lock[path] = self.rw_lockable_class(**lock_init_kwargs)

        file_lock = self.path_to_file_lock[path]

        return file_lock.gen_rlock() if is_reader else file_lock.gen_wlock()