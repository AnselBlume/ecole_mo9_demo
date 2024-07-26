class Lockable:
    def acquire(self, *args, **kwargs) -> None:
        '''
            Acquire the lock
        '''
        raise NotImplementedError('acquire must be implemented by subclasses')

    def release(self, *args, **kwargs) -> None:
        '''
            Release the lock
        '''
        raise NotImplementedError('release must be implemented by subclasses')