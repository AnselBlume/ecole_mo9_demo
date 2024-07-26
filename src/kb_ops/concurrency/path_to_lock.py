from __future__ import annotations
from .lockable import Lockable
from .lock_generation import LockGenerator, FileLockGenerator, ReadersWritersLockGenerator

def get_path_to_lock_mapping(paths: list[str], lock_generator: LockGenerator, is_reader: bool = None) -> PathToLockMapping:
    if isinstance(lock_generator, FileLockGenerator):
        return _LockFileLockGeneratorWrapper(lock_generator)

    elif isinstance(lock_generator, ReadersWritersLockGenerator):
        return _DictWrapper({
            path : lock_generator.get_lock(path, is_reader)
            for path in paths
        })

    else:
        raise ValueError(f'Unsupported lock generator type: {type(lock_generator)}')

class PathToLockMapping:
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

class _DictWrapper(PathToLockMapping):
    def __init__(self, path_to_lock: dict[str, Lockable]):
        self.path_to_lock = path_to_lock

    def get_lock(self, path: str) -> Lockable:
        return self.path_to_lock[path]

    def __contains__(self, path: str) -> bool:
        return path in self.path_to_lock

class _LockFileLockGeneratorWrapper(PathToLockMapping):
    def __init__(self, lock_generator: LockGenerator):
        self.lock_generator = lock_generator

    def get_lock(self, path: str) -> Lockable:
        return self.lock_generator.get_lock(path + '.lock')

    def __contains__(self, path: str) -> bool:
        return True