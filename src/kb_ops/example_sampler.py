from model.concept import ConceptKB, Concept, ConceptExample
import numpy as np
import logging
from typing import Literal

logger = logging.getLogger(__file__)

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
        max_to_sample_per_descendant: int = None, # TODO
        negatives_strategy: Literal['use_all_concepts_as_negatives', 'use_siblings_as_negatives', 'only_positives'] = 'use_siblings_as_negatives'
    ):
        '''
            use_descendants_as_positives, aka train ancestors on positives.
        '''
        assert all(example.concept_name for example in concept_examples), 'All examples must have a concept name'

        if negatives_strategy == 'use_all_concepts_as_negatives':
            concepts_to_train_per_example = [None for _ in range(len(concept_examples))] # Each example should train on all concepts

        elif negatives_strategy == 'use_siblings_as_negatives':
            def get_siblings_names(concept: Concept) -> list[Concept]:
                # Include the current concept in the returned sibling list. The current concept IS NOT
                # guaranteed to be first in the list
                siblings = {concept.name} # Prevent duplicates
                for parent in concept.parent_concepts.values():
                    siblings.update(dict.fromkeys(parent.child_concepts.keys())) # Store the sibling names

                return list(siblings)

            concept_to_siblings = {
                concept_name : get_siblings_names(self.concept_kb[concept_name])
                for concept_name in dict.fromkeys([example.concept_name for example in concept_examples])
            }

            concepts_to_train_per_example = [
                concept_to_siblings[example.concept_name]
                for example in concept_examples
            ]

        elif negatives_strategy == 'only_positives':
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
