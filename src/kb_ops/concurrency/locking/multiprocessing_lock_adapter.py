import multiprocessing as mp
from typing import Any

class ThreadingToMultiprocessingLockAdapter:
    '''
        Wrapper for multiprocessing.Lock to make its acquire method's "block" kwarg compatible with
        the threading.Lock acquire's method interface with its "blocking" kwarg.

        Can't easily subclass mp.Lock since it's actually a function instantiated at runtime.
    '''
    def __init__(self):
        # All accesses to __lock will result in the following string due to name mangling
        # This is a shortcut to avoid having to write out the entire mangled name
        super().__setattr__('_name_mangled_lock', mp.Lock()) # Avoid name collisions with mp.Lock's attributes

    def acquire(self, blocking=True, timeout=None):
        '''
            Maps the "blocking" kwarg to the "block" kwarg of multiprocessing.Lock.acquire.
        '''

        # threading's acquire expects timeout=-1 for blocking but multiprocessing expects timeout=None
        if timeout and timeout < 0:
            timeout = None

        return self._name_mangled_lock.acquire(block=blocking, timeout=timeout)

    def __getattr__(self, name):
        '''
            Forward any attribute not found by __getattribute__ to the multiprocessing.Lock.
        '''
        return getattr(super().__getattribute__('_name_mangled_lock'), name)

    def __setattr__(self, name: str, value: Any) -> None:
        '''
            Forward all attribute setting to the multiprocessing.Lock object.
        '''
        setattr(super().__getattribute__('_name_mangled_lock'), name, value)

    # Tell pickle how to pickle this object properly
    def __getstate__(self):
        '''
            Return the state of the object for pickling.
        '''
        return self._name_mangled_lock

    def __setstate__(self, state):
        '''
            Restore the state of the object from unpickling.
        '''
        super().__setattr__('_name_mangled_lock', state)