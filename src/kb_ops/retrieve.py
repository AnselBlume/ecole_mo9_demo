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

        # Compute text embeddings and cache them so they dont have to be recomputed every time
        # the clip_model will probably be on a GPU, but for now let's store the computed embeds on the CPU

        # Cache path provided and exists on disk, so load it
        if cache_path:
            raise NotImplementedError('Need to save more than just the FAISS index (e.g. concepts, mapping)')

        if cache_path and os.path.exists(self.cache_path):
            logger.info(f"[ Loading cached FAISS index from {self.cache_path} ]")
            self._index = torch.load(self.cache_path)

        # No cache path specified, or cache doesn't exist yet. Compute index and possibly persist
        else:
            self.rebuild_index()

    def _normalize_name(self, name: str):
        return name.lower().strip()

    def _get_text_embeds(self, queries: list[str]):
        inputs = self.clip_processor(text=queries, images=None, return_tensors='pt', padding=True)

        with torch.inference_mode():
            embeds = self.clip_model.get_text_features(**to_device(inputs, self.clip_model.device))

        embeds = embeds.cpu().numpy()
        faiss.normalize_L2(embeds)

        return embeds

    def rebuild_index(self):
        if not self._concepts:
            return

        logger.info('[ Computing text embeddings for FAISS index ]')
        concept_names = [self._normalize_name(c.name) for c in self._concepts]

        self.id_counter = 0
        ids = np.arange(self.id_counter, self.id_counter + len(concept_names))
        self.id_counter += len(concept_names)
        self._concept_lbl2id = {name : id for name, id in zip(concept_names, ids)}

        # Construct FAISS index
        # Flat indexes don't support add_with_index, but wrapping in IndexIDMap allows it to:
        # https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
        logger.info('[ Building FAISS index ]')

        text_embeds = self._get_text_embeds(concept_names)
        embed_dim = text_embeds.shape[1]
        self._index = faiss.IndexIDMap(faiss.IndexFlatL2(embed_dim))

        self._index.add_with_ids(text_embeds, ids)

        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            torch.save(self._index, self.cache_path)  # Cache index to 'cache_path'

    def retrieve(self, query: str, top_k: int) -> List[RetrievedConcept]:
        '''
            Returns the Concepts in decreasing order of match scores (increasing distance) based on
            the query and the Concepts' names.
        '''
        # FAISS returns the requested number of elements, just with a randomly returned element if the number
        # requested is > index size, but with distance of a giant number
        if top_k > len(self._concepts):
            logger.warning(f"[ Requested top_k ({top_k}) > index size ({len(self._concepts)}). Setting top_k to {len(self._concepts)} ]")
            top_k = len(self._concepts)

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
        if not self._concepts:
            self._concepts.append(concept)
            self.rebuild_index()
            return

        # See https://github.com/facebookresearch/faiss/wiki/Getting-started
        # Add to faiss index (and concept list)
        name = self._normalize_name(concept.name)

        id = self.id_counter
        self.id_counter += 1

        # Update concept storage and metadata
        self._concepts.append(concept)
        self._concept_lbl2id[name] = id

        # Update FAISS index
        embeds = self._get_text_embeds([name])
        self._index.add_with_ids(embeds, np.array([id]))

        logger.info(f"[ {concept.name} added to FAISS index ]")

        if update_cache:
            if not self.cache_path:
                raise ValueError("Cache path not provided. Cannot update cache.")

            torch.save(self._index, self.cache_path)
            logger.info(f"[ Added new concept ({concept.name}) into {self.cache_path}]")

    def remove_concept(self, concept_name: str, update_cache=False):
        # See https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes#removing-elements-from-an-index
        # Remove from faiss index (and concept list)
        name = self._normalize_name(concept_name)

        rid = self._concept_lbl2id[name]
        rid = np.array([rid], dtype=np.int64)

        # Remove from FAISS index
        n_removed = self._index.remove_ids(rid)
        if n_removed != 1:
            raise RuntimeError(f'Expected to remove 1 element, but removed {n_removed}')

        # Remove from concept storage and metadata
        idx = None
        for i, concept in enumerate(self._concepts):
            if self._normalize_name(concept.name) == name:
                idx = i
                break

        assert idx is not None, f'Concept {name} not found in concept list; something is seriously wrong'

        del self._concepts[idx]
        del self._concept_lbl2id[name]

        logger.info(f"[ {concept_name} removed from FAISS index ]")

        if update_cache:
            if not self.cache_path:
                raise ValueError("Cache path not provided. Cannot update cache.")

            torch.save(self._index, self.cache_path)
            logger.info(f"[ Removed concept ({name}) from {self.cache_path}]")

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
    concepts = [Concept(name=name) for name in concept_names]

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
