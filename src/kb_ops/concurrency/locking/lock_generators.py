from __future__ import annotations
from .lockable import Lockable
from .multiprocessing_lock_adapter import ThreadingToMultiprocessingLockAdapter
from .path_to_lock import PathToLockMapping
from .path_to_lock_wrappers import DictWrapper, LockGeneratorWrapper
from .lock_generator import LockGenerator
from filelock import FileLock
from readerwriterlock.rwlock import RWLockFair, RWLockable
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

class ReadersWritersLockGenerator(LockGenerator):
    def __init__(self, rw_lockable_class: RWLockable = RWLockFair, lock_factory=ThreadingToMultiprocessingLockAdapter, **lock_init_kwargs):
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

    def get_path_to_lock_mapping(self, paths: list[str], is_reader: bool = None) -> PathToLockMapping:
        return DictWrapper({
            path : self.get_lock(path, is_reader=is_reader)
            for path in paths
        })

class FileLockGenerator(LockGenerator):
    def get_lock(self, path: str, is_reader: bool = None, **lock_init_kwargs_override) -> Lockable:
        lock_file_path = path + '.lock' # Lock file path is the path with '.lock' appended to it.
        lock_init_kwargs = self._consolidate_lock_init_kwargs(lock_init_kwargs_override)
        return FileLock(lock_file_path, **lock_init_kwargs)

    def get_path_to_lock_mapping(self, paths: list[str], is_reader: bool = None) -> PathToLockMapping:
        return LockGeneratorWrapper(self)