# %%
from model.concept import Concept
from transformers import CLIPProcessor, CLIPModel
from dataclasses import dataclass
from typing import List
from utils import to_device

import numpy as np
import faiss
import torch
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class RetrievedConcept:
    concept: Concept = None
    distance: float = None

class CLIPConceptRetriever:
    def __init__(
        self,
        concepts: List[Concept],
        clip_model: CLIPModel,
        clip_processor: CLIPProcessor,
        cache_path: str = None
    ):
        self._concepts = concepts
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.cache_path = cache_path
        self._index = None
        self._concept_idx2lbl = {idx: concept_lbl.name.lower().strip() for idx, concept_lbl in enumerate(concepts)}
        self._concept_lbl2idx = {concept_lbl.name.lower().strip(): idx for idx, concept_lbl in enumerate(concepts)}

        # Compute text embeddings and cache them so they dont have to be recomputed every time
        # the clip_model will probably be on a GPU, but for now let's store the computed embeds on the CPU

        # Cache path provided and exists on disk, so load it
        if cache_path and os.path.exists(self.cache_path):
            logger.info(f"[ Loading cached FAISS index from {self.cache_path} ]")
            self._index = torch.load(self.cache_path)

        # No cache path specified, or cache doesn't exist yet. Compute index and possibly persist
        else:
            logger.info('[ Computing text embeddings for FAISS index ]')
            concept_names = [c.name.lower().strip() for c in concepts]
            text_embeds = self._get_text_embeds(concept_names)  # Convert to 'numpy.ndarray' to process with faiss.normalize_L2

            # Construct FAISS index
            logger.info('[ Building FAISS index ]')
            embed_dim = text_embeds.shape[1]
            self._index = faiss.IndexFlatL2(embed_dim)
            faiss.normalize_L2(text_embeds)
            self._index.add(text_embeds)

            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(self._index, self.cache_path)  # Cache index to 'cache_path'

    def _get_text_embeds(self, queries: list[str]):
        inputs = self.clip_processor(text=queries, images=None, return_tensors='pt', padding=True)

        with torch.inference_mode():
            outputs = self.clip_model.get_text_features(**to_device(inputs, self.clip_model.device))

        return outputs.cpu().numpy()

    def retrieve(self, query: str, top_k: int) -> List[RetrievedConcept]:
        '''
            Returns the Concepts in decreasing order of match scores (increasing distance) based on
            the query and the Concepts' names.
        '''
        # Create a search vector and search for 'top_k'
        _query_embeds = self._get_text_embeds([query])
        faiss.normalize_L2(_query_embeds)
        distances, nn = self._index.search(_query_embeds, k=top_k) # Each is of shape (1, top_k) as only one query

        retrieved_concepts = [
            RetrievedConcept(concept=self._concepts[concept_idx], distance=dist)
            for concept_idx, dist in zip(nn[0], distances[0])
        ]

        return retrieved_concepts

    def add_concept(self, concept: Concept, update_cache=False):
        # Add to faiss index (and concept list)
        inputs = self.clip_processor(text=[concept.name], images=None, return_tensors='pt', padding=True)
        outputs = self.clip_model(**inputs)
        self.concepts.append(concept)  # Update the 'self._concepts' list
        self._index.add(outputs.text_embeds.detach().numpy())  # Update the faiss index

        if update_cache:
            if not self.cache_path:
                raise ValueError("Cache path not provided. Cannot update cache.")

            self._concept_idx2lbl[len(self._concept_idx2lbl)] = concept.name.lower().strip()
            self._concept_lbl2idx[concept.name.lower().strip()] = len(self._concept_lbl2idx)
            torch.save(self._index, self.cache_path)
            logger.info(f"[ Added new concept ({concept.name}) into {self.cache_path}]")

    def remove_concept(self, concept_name: str, update_cache=False):
        # Remove from faiss index (and concept list)
        rid = self._concept_lbl2idx[concept_name.lower().strip()]
        rid = np.array([rid], dtype=np.int64)
        self._index.remove_ids(rid)
        logger.info(f"[ {concept_name} removed from FAISS index ]")

        if update_cache:
            if not self.cache_path:
                raise ValueError("Cache path not provided. Cannot update cache.")

            self._concept_idx2lbl[len(self._concept_idx2lbl)] = concept.name.lower().strip()
            self._concept_lbl2idx[concept.name.lower().strip()] = len(self._concept_lbl2idx)
            torch.save(self._index, self.cache_path)
            logger.info(f"[ Removed concept ({concept.name}) from {self.cache_path}]")

# %%
if __name__ == "__main__":
    '''
    Running 'python retrieve.py' will return a list of top-k concepts according to the given query input
    '''
    from feature_extraction import build_clip
    import coloredlogs
    coloredlogs.install(logging.INFO)

    # %%
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    concept_names = [
        "Spoon", "Knife", "Hammer", "Fork", "Spatula", "Tongs", "Screwdriver", "Wrench", "Ladle", "Peeler"
    ]
    concepts = []
    for cname in concept_names:
        concept = Concept()
        concept.name = cname
        concepts.append(concept)

    clip_model, clip_processor = build_clip(device=device)
    retriever = CLIPConceptRetriever(concepts,
                                     clip_model=clip_model,
                                     clip_processor=clip_processor)

    # %%
    while True:
        k = 3
        query = input("Enter a query string >> ")
        if query.lower().strip() in ['exit', '']:
            break

        retrieved = retriever.retrieve(query, top_k=k)
        logger.info(f"[ Top-K concepts retrieved (Query :: {query}) ]")
        logger.info([f"{r.concept.name} ({r.distance:.2f})" for r in retrieved])
# %%
