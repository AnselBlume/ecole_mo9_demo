from __future__ import annotations
from .lockable import Lockable
from .lock_generator import LockGenerator
from .path_to_lock import PathToLockMapping

class DictWrapper(PathToLockMapping):
    def __init__(self, path_to_lock: dict[str, Lockable]):
        self.path_to_lock = path_to_lock

    def get_lock(self, path: str) -> Lockable:
        return self.path_to_lock[path]

    def __contains__(self, path: str) -> bool:
        return path in self.path_to_lock

class LockGeneratorWrapper(PathToLockMapping):
    def __init__(self, lock_generator: LockGenerator):
        self.lock_generator = lock_generator

    def get_lock(self, path: str) -> Lockable:
        return self.lock_generator.get_lock(path)

    def __contains__(self, path: str) -> bool:
        return True