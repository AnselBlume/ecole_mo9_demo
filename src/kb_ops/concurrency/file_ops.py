from .locking import Lockable, PathToLockMapping
from typing import Callable, Any, IO
import pickle

def load_pickle(path: str, lock: Lockable = None, path_to_lock: PathToLockMapping = None) -> Any:
    '''
        Loads an object from a file at the given path using pickle.
        If a lock is provided, it will be acquired before the operation and released after.
        If path_to_lock is provided, the lock will be retrieved from the mapping.
    '''
    return exec_file_op(
        path,
        file_open_mode='rb',
        operation=pickle.load,
        lock=lock,
        path_to_lock=path_to_lock
    )

def dump_pickle(obj: Any, path: str, lock: Lockable = None, path_to_lock: PathToLockMapping = None) -> None:
    '''
        Dumps an object to a file at the given path using pickle.
        If a lock is provided, it will be acquired before the operation and released after.
        If a path_to_lock is provided, the lock will be retrieved from the mapping.
    '''
    return exec_file_op(
        path,
        file_open_mode='wb',
        operation=lambda f: pickle.dump(obj, f),
        lock=lock,
        path_to_lock=path_to_lock
    )

def exec_file_op(
    path: str,
    file_open_mode: str = 'rb',
    operation: Callable[[IO], Any] = pickle.load,
    lock: Lockable = None,
    path_to_lock: PathToLockMapping = None
) -> Any:
    '''
        Executes an operation on a file at the given path, with the given file options.
        If a lock is provided, it will be acquired before the operation and released after.
        If path_to_lock is provided, the lock will be retrieved from the mapping.

        Args:
            path (str): Path to the file
            file_open_mode (str): Mode to open the file with
            operation (Callable[[IO], Any]): Operation to perform on the file
            lock (Lockable): Lock to use. Takes precedence over path_to_lock
            path_to_lock (PathToLockMapping): Mapping from paths to locks

        Returns:
            Any: Result of the operation
    '''

    if lock: # Use provided lock
        if path_to_lock:
            raise RuntimeError('Cannot specify both lock and path_to_lock')

    elif path_to_lock: # Attempt to get lock from path_to_lock
        if path not in path_to_lock:
            raise RuntimeError(f'Path {path} not in path_to_lock')

        lock = path_to_lock.get_lock(path)

    else: # No lock provided
        lock = None

    if lock:
        lock.acquire()

    with open(path, file_open_mode) as f:
        result = operation(f)

    if lock:
        lock.release()

    return result