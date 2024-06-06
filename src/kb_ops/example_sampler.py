from model.concept import ConceptKB, Concept, ConceptExample
import numpy as np
import logging
from typing import Union, Literal
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__file__)

class ConceptsToTrainNegativeStrategy(Enum):
    '''
        See ExampleSampler.get_concepts_to_train_per_example for more information on each strategy.
    '''
    use_all_concepts_as_negatives = 'use_all_concepts_as_negatives'
    use_siblings_as_negatives = 'use_siblings_as_negatives'
    only_positives = 'only_positives'

@dataclass
class ConceptKBExampleSamplerConfig:
    use_descendants_as_positives: bool = True
    use_containing_concepts_for_positives: bool = True
    negatives_strategy: ConceptsToTrainNegativeStrategy = ConceptsToTrainNegativeStrategy.use_siblings_as_negatives

    random_seed: int = 42

class ConceptKBExampleSampler:
    def __init__(
        self,
        concept_kb: ConceptKB,
        config: ConceptKBExampleSamplerConfig = ConceptKBExampleSamplerConfig()
    ):
        self.concept_kb = concept_kb
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def get_all_examples(
        self,
        concepts: list[Concept],
        include_concept_specific_negatives_for: Union[None, Literal['all'], list[str]] = None
    ) -> tuple[list[ConceptExample], list[str]]:

        '''
            include_concept_specific_negatives_for:
                - None: Only include positive examples
                - 'all': Include all examples
                - list of str: Include negative examples for the specified concepts (each string is a concept name)
        '''
        # Construct set of concepts to include negatives for
        if not include_concept_specific_negatives_for:
            concepts_to_include_negatives_for = set()

        elif include_concept_specific_negatives_for == 'all':
            concepts_to_include_negatives_for = {concept.name for concept in concepts}

        elif isinstance(include_concept_specific_negatives_for, list): # Guaranteed to be > 0 since not Falsey
            assert all(isinstance(elmt, str) for elmt in include_concept_specific_negatives_for)
            concepts_to_include_negatives_for = set(include_concept_specific_negatives_for)

        else:
            raise ValueError(f'Invalid include_concept_specific_negatives_for: {include_concept_specific_negatives_for}')

        # Get all examples for the specified concepts
        all_examples = []
        all_labels = []
        for concept in concepts:
            examples = concept.examples

            if concept.name not in concepts_to_include_negatives_for:
                examples = [example for example in examples if not example.is_negative]

            n_examples = len(examples)
            all_examples.extend(examples)
            all_labels.extend([concept.name] * n_examples)

        return all_examples, all_labels

    def sample_examples(
        self,
        concepts: list[Concept],
        n_examples_per_concept: int = None,
        n_examples_from_union: int = None,
        include_concept_specific_negatives_for: Union[None, Literal['all'], list[str]] = None
    ) -> tuple[list[ConceptExample], list[str]]:
        '''
            If n_examples_per_concept, samples n_examples_per_concept examples from each concept in concepts.
            If n_examples_from_union, samples n_examples_from_union examples from the union of all concepts' examples.
        '''
        if not (bool (n_examples_per_concept) ^ bool(n_examples_from_union)):
            raise ValueError('Exactly one of n_examples_per_concept or n_examples_from_union must be specified')

        if n_examples_from_union:
            # Sample n_examples_from_union examples from the union of all concepts
            all_examples, all_labels = self.get_all_examples(concepts, include_concept_specific_negatives_for=include_concept_specific_negatives_for)

            try:
                sampled_indices = self.rng.choice(len(all_examples), n_examples_from_union, replace=False)
            except ValueError:
                logger.debug(f'Not enough examples to sample from for all concepts; using all examples')
                sampled_indices = range(len(all_examples))

            sampled_examples = [all_examples[i] for i in sampled_indices]
            sampled_labels = [all_labels[i] for i in sampled_indices]

        if n_examples_per_concept:
            sampled_examples = []
            sampled_labels = []
            for concept in concepts:
                examples, _ = self.get_all_examples([concept], include_concept_specific_negatives_for=include_concept_specific_negatives_for)

                try:
                    examples = self.rng.choice(examples, n_examples_per_concept, replace=False)
                except ValueError:
                    logger.debug(f'Not enough examples to sample from for concept {concept.name}; using all examples')

                sampled_examples.extend(examples)
                sampled_labels.extend([concept.name] * len(examples))

        return sampled_examples, sampled_labels

    def sample_negative_examples(
        self,
        n_pos_examples: int,
        neg_concepts: list[Concept],
        min_neg_ratio_per_concept: float = 1.0,
        sample_from_descendants: bool = True
    ) -> tuple[list[ConceptExample], list[str]]:
        '''
            Samples negative examples from the given negative concepts, trying to match the given ratio.

            Arguments:
                n_pos_examples: Number of positive examples
                neg_concepts: List of negative concepts to sample at least one example of each from
                min_neg_ratio: Minimum ratio of negative examples to positive examples
                sample_from_descendants: If True, samples negatives from the union of a concept and its descendants' examples

            Returns: Tuple of (sampled_examples, sampled_concept_names)
        '''
        # Decide how many negatives to sample per concept
        n_neg_per_concept = max(int(min_neg_ratio_per_concept * n_pos_examples), 1)

        logger.info(f'Attempting to sample {n_neg_per_concept} negative examples per concept')

        sampled_examples = []
        sampled_concept_names = []
        for neg_concept in neg_concepts:

            if sample_from_descendants:
                examples, labels = self.get_all_examples(self.concept_kb.rooted_subtree(neg_concept))
            else:
                examples, labels = neg_concept.examples, [neg_concept.name] * len(neg_concept.examples)

            try:
                sampled_inds = self.rng.choice(len(examples), n_neg_per_concept, replace=False)
                neg_examples = [examples[i] for i in sampled_inds]
                neg_labels = [labels[i] for i in sampled_inds]

            except ValueError: # Not enough negative examples
                logger.debug(f'Not enough examples to sample from for concept {neg_concept.name}; using all examples')
                neg_examples = examples
                neg_labels = labels

            sampled_examples.extend(neg_examples)
            sampled_concept_names.extend(neg_labels)

        return sampled_examples, sampled_concept_names

    def get_concepts_to_train_per_example(
        self,
        concept_examples: list[ConceptExample],
        use_descendants_as_positives: bool = None,
        use_containing_concepts_for_positives: bool = None,
        max_to_sample_per_descendant: int = None, # TODO
        negatives_strategy: ConceptsToTrainNegativeStrategy = None
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
        if use_descendants_as_positives is None:
            use_descendants_as_positives = self.config.use_descendants_as_positives
        if use_containing_concepts_for_positives is None:
            use_containing_concepts_for_positives = self.config.use_containing_concepts_for_positives
        if negatives_strategy is None:
            negatives_strategy = self.config.negatives_strategy

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

        if use_containing_concepts_for_positives:
            # To use containing concepts for positives, we need to add component concepts to the list of concepts to train on
            for example, concepts_to_train in zip(concept_examples, concepts_to_train_per_example):
                concepts_to_train_set = set(concepts_to_train)

                for component_name in self.concept_kb[example.concept_name].component_concepts:
                    if component_name not in concepts_to_train_set:
                        concepts_to_train.append(component_name)

        return concepts_to_train_per_example
