from base import BaseController
from kb_ops.dataset import split_from_concept_kb
from model.concept import Concept, ConceptExample
from typing import Literal
import logging

logger = logging.getLogger(__file__)

class ControllerTrainMixin(BaseController):

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
                    self.cacher.recache_zs_attr_features(component, examples=examples) # Needed to predict the componnt concept

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
        **train_concept_kwargs
    ):
        # Ensure features are prepared, only generating those which don't already exist or are dirty
        # Cache all concepts, since we might sample from concepts whose examples haven't been cached yet
        self.cacher.cache_segmentations(only_uncached_or_dirty=True)
        self.cacher.cache_features(only_uncached_or_dirty=True)

        # TODO Add a variant that merges all of the datasets (merging duplicate examples using datasets' concepts_to_train field) and runs trainer.train()

        concepts = [self.retrieve_concept(c, max_retrieval_distance=max_retrieval_distance) for c in concept_names]

        dependencies = self._get_concept_retraining_dependencies(concepts)
        dependencies = {k : v for k, v in sorted(dependencies.items(), key=lambda item: len(item[1]))} # Sort in ascending order of number of dependencies

        was_trained = set()

        for concept, concept_dependencies in dependencies.items():
            # NOTE this if statement is a nop as this is single-threaded, but demonstrates what to check for in multiprocessing
            concept_dependencies_to_be_trained = set(concept_dependencies).intersection(concepts)

            if not all(dependency in was_trained for dependency in concept_dependencies_to_be_trained): # All dependencies should have been trained
                raise RuntimeError(f'Concept "{concept.name}" has untrained dependencies: {concept_dependencies}')

            examples, dataset = self.trainer.construct_dataset_for_concept_training(concept, use_concepts_as_negatives=use_concepts_as_negatives)

            # Recache zero-shot attributes for sampled examples
            self.cacher.recache_zs_attr_features(concept, examples=examples)

            if self.use_concept_predictors_for_concept_components:
                for component in concept.component_concepts.values():
                    self.cacher.recache_zs_attr_features(component, examples=examples) # Needed to predict the componnt concept

            else: # Using fixed scores for concept-image pairs
                self.cacher.recache_component_concept_scores(concept, examples=examples)

            self.trainer.train_concept(concept, samples_and_dataset=(examples, dataset), n_epochs=n_epochs, **train_concept_kwargs)
            was_trained.add(concept)

    # Get concepts to train and each of their dependencies
    # For each concept, get its dataset and examples, recaching each.
    # Then call trainer.train_concept
    def _get_concepts_to_train_to_update_concept(
        self,
        concept: Concept,
        include_siblings: bool = False,
        include_ancestors: bool = False,
        include_ancestors_siblings: bool = False,
        include_component_roots_as_ancestor_siblings: bool = False,
    ) -> list[Concept]:
        '''
            Computes the set of concepts to train if the intent is to update a concept's predictor.
            I.e., the set of concepts that would be influenced by the training of the specified concept.

            Keyword arguments determine the comprehensiveness of the training process, with more set to True
            resulting in a larger set of concepts to train. By default, only the specified concept is trained.

            include_siblings: If True, includes the siblings of the concepts in the set of concepts to train.
            include_ancestors: If True, includes the ancestors of the concepts in the training set.
            include_ancestors_siblings: If True, includes the siblings of the ancestors of the concepts in the training set.
            include_component_roots_as_ancestor_siblings: If True, includes component root concepts as ancestor siblings
                (as opposed to just non-component root Concepts) in the concepts in the training set.
        '''
        concepts_to_train = {concept : None}

        if include_siblings:
            for parent in concept.parent_concepts.values():
                concepts_to_train.update(dict.fromkeys(parent.child_concepts.values())) # Includes self, but that's fine

        if include_ancestors:
            ancestors = self.concept_kb.rooted_subtree(concept, reverse_edges=True) # Includes self, but that's fine
            concepts_to_train.update(dict.fromkeys(ancestors))

        if include_ancestors_siblings:
            if 'ancestors' not in locals():
                ancestors = self.concept_kb.rooted_subtree(concept, reverse_edges=True)

            concepts_to_train.update(dict.fromkeys(self.concept_kb.children_of([a for a in ancestors if a != concept])))

            # Root nodes are also "siblings" at the highest level
            if include_component_roots_as_ancestor_siblings:
                component_concepts = set(self.concept_kb.component_concepts)
                concepts_to_train.update({c : None for c in self.concept_kb.root_concepts if c not in component_concepts})
            else:
                concepts_to_train.update(dict.fromkeys(self.concept_kb.root_concepts))

        return list(concepts_to_train)

    def _get_concept_retraining_dependencies(self, concepts_to_train: list[Concept]) -> dict[Concept, list[Concept]]:
        '''
            Returns a mapping from concepts (names) to concepts (names) they are dependent on for training/inference.

            dependencies[concept] = [Concept A, Concept B, ...] means Concept A, Concept B, ... should be retrained before Concept
            (if they are to be retrained).
        '''
        return {
            concept : list(concept.component_concepts.values())
            for concept in concepts_to_train
        }