from readerwriterlock.rwlock import Lockable
from typing import Callable, Any, IO
import pickle

def load_pickle(path: str, lock: Lockable = None, path_to_lock: dict[str, Lockable] = {}) -> Any:
    '''
        Loads an object from a file at the given path using pickle.
        If a lock is provided, it will be acquired before the operation and released after.
        If a path_to_lock dictionary is provided, the lock will be retrieved from the dictionary.
    '''
    return exec_file_op(
        path,
        file_options='rb',
        operation=pickle.load,
        lock=lock,
        path_to_lock=path_to_lock
    )

def dump_pickle(obj: Any, path: str, lock: Lockable = None, path_to_lock: dict[str, Lockable] = {}) -> None:
    '''
        Dumps an object to a file at the given path using pickle.
        If a lock is provided, it will be acquired before the operation and released after.
        If a path_to_lock dictionary is provided, the lock will be retrieved from the dictionary.
    '''
    return exec_file_op(
        path,
        file_options='wb',
        operation=lambda f: pickle.dump(obj, f),
        lock=lock,
        path_to_lock=path_to_lock
    )

def exec_file_op(
    path: str,
    file_options: str = 'rb',
    operation: Callable[[IO], Any] = pickle.load,
    lock: Lockable = None,
    path_to_lock: dict[str, Lockable] = {}
) -> Any:
    '''
        Executes an operation on a file at the given path, with the given file options.
        If a lock is provided, it will be acquired before the operation and released after.
        If a path_to_lock dictionary is provided, the lock will be retrieved from the dictionary.

        Args:
            path (str): Path to the file
            file_options (str): Options to open the file with
            operation (Callable[[IO], Any]): Operation to perform on the file
            lock (Lockable): Lock to use. Takes precedence over path_to_lock
            path_to_lock (dict[str, Lockable]): Dictionary mapping paths to locks

        Returns:
            Any: Result of the operation
    '''

    if lock: # Use provided lock
        if path_to_lock:
            raise RuntimeError('Cannot specify both lock and path_to_lock')

    elif path_to_lock: # Attempt to get lock from path_to_lock
        if path not in path_to_lock:
            raise RuntimeError(f'Path {path} not in path_to_lock')

        lock = path_to_lock[path]

    else: # No lock provided
        lock = None

    if lock:
        lock.acquire()

    with open(path, file_options) as f:
        result = operation(f)

    if lock:
        lock.release()

    return result