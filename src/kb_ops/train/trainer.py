from .batched_trainer import ConceptKBBatchedTrainerMixin
from .sgd_trainer import ConceptKBSGDTrainerMixin

class ConceptKBTrainer(
    ConceptKBSGDTrainerMixin,
    ConceptKBBatchedTrainerMixin
):
    pass