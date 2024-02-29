import sys
# TODO Remove / revise the path to src in the line below
sys.path.append('/home/jk100/code/ecole_mo9_demo/src')

from model.concept import Concept
from transformers import CLIPProcessor, CLIPModel, CLIPTextModelWithProjection
from dataclasses import dataclass
from typing import List

import numpy as np
import faiss
import torch
import os

@dataclass
class RetrievedConcept:
    concept: Concept = None
    distance: float = None

class CLIPConceptRetriever:
    def __init__(self, concepts: List[Concept], clip_model: CLIPModel, clip_processor: CLIPProcessor, device='cpu'):
        self._concepts = concepts
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self._index = None
        self._concept_idx2lbl = {idx: concept_lbl.name.lower().strip() for idx, concept_lbl in enumerate(concepts)}
        self._concept_lbl2idx = {concept_lbl.name.lower().strip(): idx for idx, concept_lbl in enumerate(concepts)}

        # Compute text embeddings and cache them so they dont have to be recomputed every time
        # the clip_model will probably be on a GPU, but for now let's store the computed embeds on the CPU
        _cache_dir = f"cache_faiss"
        self._cache_path = os.path.join(_cache_dir, f"demo_concepts_index_cached.pth")  # TODO: You can change the file name to suit your needs

        if not os.path.exists(_cache_dir):
            os.mkdir(_cache_dir)

        if not os.path.exists(self._cache_path) and self._index is None:
            concept_names = [c.name.strip() for c in concepts]
            inputs = clip_processor(text=concept_names, images=None, return_tensors='pt', padding=True)
            outputs = clip_model(**inputs)
            text_embeds = outputs.text_embeds.detach().numpy()  # Convert to 'numpy.ndarray' to process with faiss.normalize_L2

            # Construct FAISS index
            embed_dim = text_embeds.shape[1]
            self._index = faiss.IndexFlatL2(embed_dim)
            faiss.normalize_L2(text_embeds)
            self._index.add(text_embeds)
            torch.save(self._index, self._cache_path)  # Cache index to 'cache_path'
        else:
            print(f"[ Loading cached FAISS index from {self._cache_path} ]")
            self._index = torch.load(self._cache_path)


    def retrieve(self, query: str, top_k: int) -> List[RetrievedConcept]:
        '''
            Returns the Concepts in decreasing order of match scores (increasing distance) based on
            the query and the Concepts' names.
        '''
        # Create a search vector and search for 'top_k'
        inputs = self.clip_processor(text=[query], images=None, return_tensors='pt', padding=True)
        outputs = self.clip_model(**inputs)
        _query_embeds = outputs.text_embeds.detach().numpy()
        faiss.normalize_L2(_query_embeds)
        distances, nn = self._index.search(_query_embeds, k=top_k)
        return distances, nn

    def add_concept(self, concept: Concept, update_cache=False):
        # Add to faiss index (and concept list)
        inputs = self.clip_processor(text=[concept.name], images=None, return_tensors='pt', padding=True)
        outputs = self.clip_model(**inputs)
        self.concepts.append(concept)  # Update the 'self._concepts' list
        self._index.add(outputs.text_embeds.detach().numpy())  # Update the faiss index
        if update_cache:
            print(f"[ Added new concept ({concept.name}) into {self._cached_path}]")
            self._concept_idx2lbl[len(self._concept_idx2lbl)] = concept.name.lower().strip()
            self._concept_lbl2idx[concept.name.lower().strip()] = len(self._concept_lbl2idx)
            torch.save(self._index, self._cache_path)

    def remove_concept(self, concept_name: str, update_cache=False):
        # Remove from faiss index (and concept list)
        rid = self._concept_lbl2idx[concept_name.lower().strip()]
        rid = np.array([rid], dtype=np.int64)
        self._index.remove_ids(rid)
        print(f"[ {concept_name} removed from FAISS index ]")
        if update_cache:
            print(f"[ Removed concept ({concept.name}) from {self._cached_path}]")
            self._concept_idx2lbl[len(self._concept_idx2lbl)] = concept.name.lower().strip()
            self._concept_lbl2idx[concept.name.lower().strip()] = len(self._concept_lbl2idx)
            torch.save(self._index, self._cache_path)


if __name__ == "__main__":
    '''
    Running 'python retrieve.py' will return a list of top-k concepts according to the given query input
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    concept_names = [
        "Spoon", "Knife", "Hammer", "Fork", "Spatula", "Tongs", "Screwdriver", "Wrench", "Ladle", "Peeler"
    ]
    query = input("Enter a query string >> ")
    concepts = []
    for cname in concept_names:
        concept = Concept()
        concept.name = cname
        concepts.append(concept)

    encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    retriever = CLIPConceptRetriever(concepts, 
                                     clip_model=encoder,
                                     clip_processor=processor)
    k = 3
    topk_distances, topk_concepts = retriever.retrieve(query, top_k=k)
    print(f"[ Top-K concepts retrieved (Query :: {query}) ]")
    print("top-k distances >> ", topk_distances)
    print("top-k concepts >> ", topk_concepts)