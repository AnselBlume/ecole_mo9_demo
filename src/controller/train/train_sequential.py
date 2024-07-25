from .base import ControllerTrainMixinBase
from kb_ops.dataset import split_from_concept_kb
from model.concept import Concept, ConceptExample
from typing import Literal
from kb_ops.concurrency import ConcurrentTrainingConceptSelector
import logging

logger = logging.getLogger(__file__)

class ControllerTrainSequentialMixin(ControllerTrainMixinBase):
    def train(
        self,
        split: tuple[float, float, float] = (.6, .2, .2),
        use_concepts_as_negatives: bool = False
    ):
        '''
            Trains all concepts in the concept knowledge base from each concept's example_imgs.
        '''

        self.cacher.cache_segmentations()
        self.cacher.cache_features()

        # Recache all concepts' zero-shot features in case new concepts were added since last training
        for concept in self.concept_kb:
            self.cacher.recache_zs_attr_features(concept)

            if not self.use_concept_predictors_for_concept_components: # Using fixed scores for concept-image pairs
                self.cacher.recache_component_concept_scores(concept)

        train_ds, val_ds, test_ds = split_from_concept_kb(self.concept_kb, split=split, use_concepts_as_negatives=use_concepts_as_negatives)

        self.trainer.train(
            train_ds=train_ds,
            val_ds=val_ds,
            n_epochs=15,
            lr=1e-2,
            backward_every_n_concepts=10,
            ckpt_dir=None
        )

    def train_concept(
        self,
        concept_name: str,
        stopping_condition: Literal['n_epochs'] = 'n_epochs',
        new_examples: list[ConceptExample] = [],
        n_epochs: int = 5,
        max_retrieval_distance=.01,
        use_concepts_as_negatives: bool = True
    ):
        '''
            Trains the specified concept with name concept_name for the specified number of epochs.

            Args:
                concept_name: The concept to train. If it does not exist, it will be created.
                stopping_condition: The condition to stop training. Must be 'n_epochs'.
                new_examples: If provided, these examples will be added to the concept's examples list.
        '''
        # Try to retrieve concept
        concept = self.retrieve_concept(concept_name, max_retrieval_distance=max_retrieval_distance) # Low retrieval distance to force exact match
        logger.info(f'Retrieved concept with name: "{concept.name}"')

        self.add_examples(new_examples, concept=concept) # Nop if no examples to add

        # Ensure features are prepared, only generating those which don't already exist or are dirty
        # Cache all concepts, since we might sample from concepts whose examples haven't been cached yet
        self.cacher.cache_segmentations(only_uncached_or_dirty=True)
        self.cacher.cache_features(only_uncached_or_dirty=True)

        # Hook to recache zs_attr_features after negative examples have been sampled
        # This is faster than calling recache_zs_attr_features on all examples in the concept_kb
        def cache_hook(examples):
            self.cacher.recache_zs_attr_features(concept, examples=examples)

            # Handle component concepts
            if self.use_concept_predictors_for_concept_components:
                for component in concept.component_concepts.values():
                    self.cacher.recache_zs_attr_features(component, examples=examples) # Necessary to predict component concepts

            else: # Using fixed scores for concept-image pairs
                self.cacher.recache_component_concept_scores(concept, examples=examples)

        if stopping_condition == 'n_epochs':
            self.trainer.train_concept(
                concept,
                stopping_condition='n_epochs',
                n_epochs=n_epochs,
                post_sampling_hook=cache_hook,
                lr=1e-2,
                dataset_construction_kwargs={
                    'use_concepts_as_negatives' : use_concepts_as_negatives
                }
            )

        else:
            raise ValueError('Unrecognized stopping condition')

    def train_concepts(
        self,
        concept_names: list[str],
        n_epochs: int = 5,
        use_concepts_as_negatives: bool = True,
        max_retrieval_distance=.01,
        concepts_to_train_kwargs: dict = {},
        **train_concept_kwargs
    ):
        # Ensure features are prepared, only generating those which don't already exist or are dirty
        # Cache all concepts, since we might sample from concepts whose examples haven't been cached yet
        self.cacher.cache_segmentations(only_uncached_or_dirty=True)
        self.cacher.cache_features(only_uncached_or_dirty=True)

        # TODO Add a variant that merges all of the datasets (merging duplicate examples using datasets' concepts_to_train field) and runs trainer.train()

        concepts = [self.retrieve_concept(c, max_retrieval_distance=max_retrieval_distance) for c in concept_names]

        concepts_to_train: dict[Concept, None] = {}
        for concept in concepts:
            concepts_to_train.update(dict.fromkeys(self._get_concepts_to_train_to_update_concept(concept, **concepts_to_train_kwargs)))

        logger.info(f'Concepts to train: {[c.name for c in concepts_to_train]}')

        concept_selector = ConcurrentTrainingConceptSelector(list(concepts_to_train))

        while not concept_selector.is_training_complete:
            concept = concept_selector.get_next_concept()

            examples, dataset = self.trainer.construct_dataset_for_concept_training(concept, use_concepts_as_negatives=use_concepts_as_negatives)

            # Recache zero-shot attributes for sampled examples
            self.cacher.recache_zs_attr_features(concept, examples=examples)

            if self.use_concept_predictors_for_concept_components:
                for component in concept.component_concepts.values():
                    self.cacher.recache_zs_attr_features(component, examples=examples) # Needed to predict the componnt concept

            else: # Using fixed scores for concept-image pairs
                self.cacher.recache_component_concept_scores(concept, examples=examples)

            self.trainer.train_concept(concept, samples_and_dataset=(examples, dataset), n_epochs=n_epochs, **train_concept_kwargs)

            concept_selector.mark_concept_completed(concept)