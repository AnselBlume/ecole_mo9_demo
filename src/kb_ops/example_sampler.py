from model.concept import ConceptKB, Concept, ConceptExample
import numpy as np
import logging
from typing import Union
from enum import Enum

logger = logging.getLogger(__file__)

class ConceptsToTrainNegativeStrategy(Enum):
    '''
        See ExampleSampler.get_concepts_to_train_per_example for more information on each strategy.
    '''
    use_all_concepts_as_negatives = 'use_all_concepts_as_negatives'
    use_siblings_as_negatives = 'use_siblings_as_negatives'
    only_positives = 'only_positives'

class ConceptKBExampleSampler:
    def __init__(
        self,
        concept_kb: ConceptKB,
        random_seed: int = 42
    ):
        self.concept_kb = concept_kb
        self.rng = np.random.default_rng(random_seed)

    def get_all_examples(self, concepts: list[Concept]) -> tuple[list[ConceptExample], list[str]]:
        all_examples = []
        all_labels = []
        for concept in concepts:
            n_examples = len(concept.examples)
            all_examples.extend(concept.examples)
            all_labels.extend([concept.name] * n_examples)

        return all_examples, all_labels

    def sample_examples(
        self,
        concepts: list[Concept],
        n_examples_per_concept: int
    ):
        '''
            Samples n_examples_per_concept examples from each concept in concepts.
        '''
        sampled_examples = []
        sampled_labels = []
        for concept in concepts:
            try:
                examples = self.rng.choice(concept.examples, n_examples_per_concept, replace=False)
            except ValueError:
                logger.debug(f'Not enough examples to sample from for concept {concept.name}; using all examples')
                examples = concept.examples

            sampled_examples.extend(examples)

            # Add labels
            sampled_labels.extend([concept.name] * len(examples))

        return sampled_examples, sampled_labels

    def sample_negative_examples(
        self,
        n_pos_examples: int,
        neg_concepts: list[Concept],
        min_neg_ratio_per_concept: float = 1.0
    ) -> tuple[list[ConceptExample], list[str]]:
        '''
            Samples negative examples from the given negative concepts, trying to match the given ratio.

            Arguments:
                n_pos_examples: Number of positive examples
                neg_concepts: List of negative concepts to sample at least one example of each from
                min_neg_ratio: Minimum ratio of negative examples to positive examples
                rng: Random number generator used for sampling from negative concepts

            Returns: Tuple of (sampled_examples, sampled_concept_names)
        '''
        # TODO Sample descendants of each negative concept if the concept is not a leaf node

        # Decide how many negatives to sample per concept
        n_neg_per_concept = max(int(min_neg_ratio_per_concept * n_pos_examples), 1)

        logger.info(f'Attempting to sample {n_neg_per_concept} negative examples per concept')

        sampled_examples = []
        sampled_concept_names = []
        for neg_concept in neg_concepts:
            try:
                neg_examples = self.rng.choice(neg_concept.examples, n_neg_per_concept, replace=False)

            except ValueError: # Not enough negative examples
                logger.debug(f'Not enough examples to sample from for concept {neg_concept.name}; using all examples')
                neg_examples = neg_concept.examples

            sampled_examples.extend(neg_examples)
            sampled_concept_names.extend([neg_concept.name] * len(neg_examples))

        return sampled_examples, sampled_concept_names

    def get_concepts_to_train_per_example(
        self,
        concept_examples: list[ConceptExample],
        use_descendants_as_positives: bool = True,
        use_containing_concepts_as_positives: bool = False, # TODO
        max_to_sample_per_descendant: int = None, # TODO
        negatives_strategy: ConceptsToTrainNegativeStrategy = ConceptsToTrainNegativeStrategy.use_siblings_as_negatives
    ) -> Union[list[None], list[list[str]]]:
        '''
            Returns the list of concepts to train per example based on the passed arguments.
            Note that this doesn't actually sample examples, but instead returns the list of concepts to train for each example

            use_descendants_as_positives: trains ancestors on positive examples.

            use_containing_concepts_as_positives: If A contains B, uses images of A as positives for B. This is not necessarily good, unless A always
                contains B, instead of sometimes contains B (or B is only sometimes visible).

            negatives_strategy:
                - use_all_concepts_as_negatives: Each example trains on all concepts. Effective for classification, but not for component detection
                  (since oftentimes we don't know whether an image contains a component or not).

                - use_siblings_as_negatives: Each example trains the concept label's siblings and all of its ancestor's siblings. This is effective for
                  downwards traversal of the tree during prediction. For components, only uses the component's siblings as it is more likely that
                  multiple components can be in an image.

                - only_positives: Only train on the positive concept for an image. This is effective for component detection, but not for classification.
        '''
        assert all(example.concept_name for example in concept_examples), 'All examples must have a concept name'

        if negatives_strategy == ConceptsToTrainNegativeStrategy.use_all_concepts_as_negatives:
            concepts_to_train_per_example = [None for _ in range(len(concept_examples))] # Each example should train on all concepts

        elif negatives_strategy == ConceptsToTrainNegativeStrategy.use_siblings_as_negatives:
            def get_siblings_names(concept: Concept) -> list[Concept]:
                # Include the current concept in the returned sibling list. The current concept IS NOT
                # guaranteed to be first in the list
                siblings = {concept.name} # Prevent duplicates

                # Get siblings of all ancestors
                ancestors = dict.fromkeys(self.concept_kb.rooted_subtree(concept, reverse_edges=True))
                for ancestor in ancestors:
                    if ancestor.name != concept.name: # This concept's children are not siblings
                        # Don't include siblings of ancestors which are also ancestors, as we only add concepts for negatives here
                        siblings_and_not_ancestors = {c.name : None for c in ancestor.child_concepts.values() if c not in ancestors}
                        siblings.update(siblings_and_not_ancestors)

                # Add component siblings only for the current concept, as it is more likely that multiple components can be in an image
                for container in concept.containing_concepts.values():
                    siblings.update(dict.fromkeys(container.component_concepts.keys()))

                return list(siblings)

            concept_to_siblings = {
                concept_name : get_siblings_names(self.concept_kb[concept_name])
                for concept_name in dict.fromkeys([example.concept_name for example in concept_examples])
            }

            concepts_to_train_per_example = [
                concept_to_siblings[example.concept_name]
                for example in concept_examples
            ]

        elif negatives_strategy == ConceptsToTrainNegativeStrategy.only_positives:
            concepts_to_train_per_example = [[example.concept_name] for example in concept_examples]

        else:
            raise ValueError(f'Invalid negatives_strategy: {negatives_strategy}')

        if use_descendants_as_positives:
            # To use descendants as positives, we need to add the ancestors to the list of concepts to train on

            # Precompute ancestors for each concept
            concept_to_ancestors: dict[str,str] = {}

            for example in concept_examples:
                if example.concept_name in concept_to_ancestors:
                    continue

                concept = self.concept_kb[example.concept_name]
                ancestors = self.concept_kb.rooted_subtree(concept, reverse_edges=True)
                concept_to_ancestors[concept.name] = [a.name for a in ancestors]

            if max_to_sample_per_descendant:
                raise NotImplementedError('max_to_sample_per_descendant is not implemented yet')

            # Add all of each concept's ancestors to its concepts_to_train list
            for example, concepts_to_train in zip(concept_examples, concepts_to_train_per_example):
                concepts_to_train_set = set(concepts_to_train)

                for ancestor in concept_to_ancestors[example.concept_name]:
                    if ancestor not in concepts_to_train_set:
                        concepts_to_train.append(ancestor)

        return concepts_to_train_per_example
