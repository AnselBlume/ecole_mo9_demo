from abc import ABC
from .lockable import Lockable
from typing import Any
from .path_to_lock import PathToLockMapping

class LockGenerator(ABC):
    def __init__(self, **lock_init_kwargs):
        self.lock_init_kwargs = lock_init_kwargs

    def get_lock(self, path: str, is_reader: bool = None, **lock_init_kwargs_override) -> Lockable:
        raise NotImplementedError('get_lock must be implemented by subclasses')

    def get_path_to_lock_mapping(self, paths: list[str], is_reader: bool = None) -> PathToLockMapping:
        raise NotImplementedError('get_path_to_lock_mapping must be implemented by subclasses')

    def _consolidate_lock_init_kwargs(self, lock_init_kwargs_override: dict) -> dict[str, Any]:
        lock_init_kwargs = self.lock_init_kwargs.copy()
        lock_init_kwargs.update(lock_init_kwargs_override)
        return lock_init_kwargs