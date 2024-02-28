from model.concept import Concept
from transformers import CLIPProcessor, CLIPModel
from dataclasses import dataclass
import faiss

@dataclass
class RetrievedConcept:
    concept: Concept = None
    distance: float = None

class CLIPConceptRetriever:
    def __init__(self, concepts: list[Concept], clip_model: CLIPModel, clip_processor: CLIPProcessor):
        self.concepts = concepts # TODO Feel free to change this to whatever data structure is best
        self.clip_model = clip_model
        self.clip_processor = clip_processor

        # TODO Compute text embeddings and cache them so they dont have to be recomputed every time
        # the clip_model will probably be on a GPU, but for now let's store the computed embeds on the CPU

        # TODO construct FAISS index

        # TODO Add whatever else you need!

    def retrieve(self, query: str, top_k: int) -> list[RetrievedConcept]:
        '''
            Returns the Concepts in decreasing order of match scores (increasing distance) based on
            the query and the Concepts' names.
        '''
        pass

    def add_concept(self, concept: Concept):
        # TODO Add to faiss index (and concept list)
        pass

    def remove_concept(self, concept_name: str):
        # TODO Remove from faiss index (and concept list)
        pass
