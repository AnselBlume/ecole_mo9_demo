from typing import Literal, Callable, Any, Optional
from kb_ops.dataset import FeatureDataset
from model.concept import Concept, ConceptExample
from typing import Union
from .outputs import TrainOutput
from .batched_trainer import ConceptKBBatchedTrainerMixin
from .sgd_trainer import ConceptKBSGDTrainerMixin
from .in_memory_trainer import ConceptKBInMemoryTrainerMixin

class ConceptKBTrainer(
    ConceptKBSGDTrainerMixin,
    ConceptKBBatchedTrainerMixin,
    ConceptKBInMemoryTrainerMixin
):
    def train_concept(
        self,
        concept: Union[str, Concept],
        *, # Force the use of kwargs after this point due to API changes
        stopping_condition: Literal['n_epochs', 'validation'] = 'n_epochs',
        n_epochs: int = 10,
        post_sampling_hook: Callable[[list[ConceptExample]], Any] = None,
        samples_and_dataset: tuple[list[ConceptExample], FeatureDataset] = None,
        dataset_construction_kwargs: dict = {},
        train_minimal: bool = True,
        **train_kwargs
    ) -> Optional[TrainOutput]:
        '''
            Trains the given concept for n_epochs.

            Arguments:
                concept: Concept to train.

                stopping_condition: One of 'n_epochs' or 'validation'.
                    If 'n_epochs', training will stop after n_epochs.

                n_epochs: Number of epochs to train for if stopping_condition is 'n_epochs'.

                post_sampling_hook: Optional hook to run after sampling examples for training.

                samples_and_dataset: Tuple of list of ConceptExamples and FeatureDataset to use for training.
                    If provided, dataset_construction_kwargs will be ignored.

                dataset_construction_kwargs: Keyword arguments to pass to construct_dataset_for_concept_training.

                train_minimal: If True, will use train_minimal method to train concept instead of train_batched.

                train_kwargs: Keyword arguments to pass to train method.
        '''
        if isinstance(concept, str):
            concept = self.concept_kb[concept]

        if id(concept) != id(self.concept_kb[concept.name]): # We train the concept from the ConceptKB, not in isolation
            raise ValueError('Concept must be from the same ConceptKB')

        if stopping_condition == 'validation':
            # Implement some way to perform validation as a stopping condition
            raise NotImplementedError('Validation is not yet implemented')

        # Train for a fixed number of epochs or until examples are predicted correctly
        else:
            if samples_and_dataset: # Use provided samples and dataset
                all_samples, train_ds = samples_and_dataset
            else: # Construct
                all_samples, train_ds = self.construct_dataset_for_concept_training(concept, **dataset_construction_kwargs)

            if post_sampling_hook:
                post_sampling_hook(all_samples)

            # Train for fixed number of epochs
            train_kwargs = train_kwargs.copy()
            train_kwargs.update({
                'concepts': [concept],
                'lr': train_kwargs.get('lr', 1e-2)
            })

            if stopping_condition == 'n_epochs':
                if train_minimal:
                    return self.train_minimal(train_ds, n_epochs, **train_kwargs)
                else:
                    train_kwargs['ckpt_dir'] = None
                    return self.train_batched(train_ds, None, n_epochs=n_epochs, **train_kwargs)

            else:
                raise ValueError(f'Unknown stopping condition: {stopping_condition}')