from score import Scorer
from concept import ConceptSet, Concept
from PIL.Image import Image

class Controller:
    def __init__(self, concept_set: ConceptSet):
        self.scorer = Scorer()
        self.concepts = concept_set

    ##############
    # Prediction #
    ##############
    def predict_image(self, image: Image):
        pass

    ###################
    # Concept Removal #
    ###################
    def clear_concepts(self):
        pass

    def remove_concept(self, concept_name: str):
        pass

    ####################
    # Concept Addition #
    ####################
    def add_concept(self, concept: Concept):
        # Get zero shot attributes (query LLM)

        # Determine if it has any obvious parent or child concepts

        # Get likely learned attributes

        # Add concept

        pass

    def _get_zs_attributes(self, concept_name: str):
        pass

    ########################
    # Concept Modification #
    ########################
    def add_zs_attribute(self, concept_name: str, zs_attr_name: str, weight: float):
        pass

    def remove_zs_attribute(self, concept_name: str, zs_attr_name: str):
        pass

    def add_learned_attribute(self, concept_name: str, learned_attr_name: str, weight: float):
        pass

    def remove_learned_attribute(self, concept_name: str, learned_attr_name: str):
        pass

    ##################
    # Interpretation #
    ##################
    def compare_concepts(self, concept1_name: str, concept2_name: str):
        pass