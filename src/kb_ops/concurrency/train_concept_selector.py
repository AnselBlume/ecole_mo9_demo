from model.concept import Concept
from collections import deque

class ConcurrentTrainingConceptSelector:
    def __init__(self, concepts_to_train: list[Concept]):
        self.concepts_to_train = concepts_to_train

        self._remaining_dependencies = self._get_initial_dependencies(concepts_to_train)
        self._leaf_concepts = deque([c for c in self._remaining_dependencies if len(self._remaining_dependencies[c]) == 0])

        self._completed_concepts = []

    @property
    def num_concepts_incomplete(self):
        return len(self.concepts_to_train) - self.num_concepts_completed

    @property
    def num_concepts_completed(self) -> int:
        return len(self._completed_concepts)

    @property
    def is_training_complete(self) -> bool:
        return self.num_concepts_completed == len(self.concepts_to_train)

    def get_next_concept(self) -> Concept:
        '''
            Returns the next concept to train, based on the current state of the training process.
            Raises IndexError if there are currently no leaf concepts to train.
        '''
        if not self.has_concept_available():
            raise IndexError('No leaf concepts available to train')

        next_concept = self._leaf_concepts.popleft()

        return next_concept

    def has_concept_available(self) -> bool:
        '''
            Returns True if there are currently leaf concepts available to train, False otherwise.
        '''
        return len(self._leaf_concepts) > 0

    def mark_concept_completed(self, concept: Concept):
        '''
            Removes the concept from the list of remaining dependencies for all concepts that depend on it (containing concepts).
        '''
        self._completed_concepts.append(concept) # Mark as completed

        # Update containing concepts
        for containing_concept in concept.containing_concepts.values():
            if containing_concept not in self.concepts_to_train:
                continue

            containing_concept_dependencies = self._remaining_dependencies[containing_concept]
            containing_concept_dependencies.remove(concept)

            if len(containing_concept_dependencies) == 0: # Concept is now a leaf concept
                self._leaf_concepts.append(containing_concept)

    def _get_initial_dependencies(self, concepts_to_train: list[Concept]) -> dict[Concept, list[Concept]]:
        concepts_to_train_set = set(concepts_to_train)

        return {
            concept :
            {component for component in concept.component_concepts.values() if component in concepts_to_train_set}
            for concept in concepts_to_train
        }