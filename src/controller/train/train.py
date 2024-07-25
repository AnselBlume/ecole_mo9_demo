from .train_sequential import ControllerTrainSequentialMixin
from .train_parallel import ControllerTrainParallelMixin

class ControllerTrainMixin(ControllerTrainSequentialMixin, ControllerTrainParallelMixin):
    pass