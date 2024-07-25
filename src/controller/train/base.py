
from controller.base import BaseController
from model.concept import Concept
import logging

logger = logging.getLogger(__file__)

class ControllerTrainMixinBase(BaseController):

    def _get_concepts_to_train_to_update_concept(
        self,
        concept: Concept,
        include_siblings: bool = True,
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
            if not concept.parent_concepts: # Root node; other roots are siblings
                root_nodes = dict.fromkeys(self.concept_kb.root_concepts)

                if not include_component_roots_as_ancestor_siblings: # Exclude component root nodes
                    component_concepts = set(self.concept_kb.component_concepts)
                    root_nodes = {c : None for c in root_nodes if c not in component_concepts}

                concepts_to_train.update(root_nodes)

            else:
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
                concepts_to_train.update(dict.fromkeys(self.concept_kb.root_concepts))
            else:
                component_concepts = set(self.concept_kb.component_concepts)
                concepts_to_train.update({c : None for c in self.concept_kb.root_concepts if c not in component_concepts})

        return list(concepts_to_train)