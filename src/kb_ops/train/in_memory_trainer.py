from .base import ConceptKBTrainerBase
import torch
import pickle
from model.concept import Concept, ConceptExample
from kb_ops.dataset import FeatureDataset
from joblib import Parallel, delayed
from typing import Any, Optional
from tqdm import tqdm
from kb_ops.dataset import BatchCachedFeaturesCollate, BatchedCachedImageFeaturesIndexer
from kb_ops.caching import CachedImageFeatures
from itertools import chain
from .outputs import TrainOutput, ValidationOutput
import logging

logger = logging.getLogger(__file__)

class ConceptKBInMemoryTrainerMixin(ConceptKBTrainerBase):
    def train_minimal_in_memory(
        self,
        train_ds: FeatureDataset,
        n_epochs: int,
        lr: float,
        concepts: list[Concept] = None,
        batch_size: int = 64,
        cache_full_dataset: bool = True
    ):
        '''
            Trains the concept predictors using the given dataset one-by-one for the specified number of epochs.
            Identical to train_minimal, but loads all features into memory before training to avoid repeated disk I/O.
        '''
        concept_to_train_dataset = self._concepts_to_datasets(train_ds, concepts=concepts)
        self.concept_kb.train()

        # Load cache
        examples_to_cache = list(self.concept_kb.global_negatives) # Copy to avoid modifying global_negatives

        if cache_full_dataset:
            concept_examples = list(chain.from_iterable([c.examples for c in self.concept_kb]))
            examples_to_cache.extend(concept_examples)

        cache: dict[str,CachedImageFeatures] = self._load_cache(examples_to_cache)

        # Train each concept
        prog_bar = tqdm(concept_to_train_dataset.items())
        for concept, dataset in prog_bar:
            prog_bar.set_description(f'Training concept "{concept.name}"')

            optimizer = torch.optim.Adam(concept.predictor.parameters(), lr=lr)
            all_features, all_labels = self._load_all_features(dataset, cache=cache)
            features_indexer = BatchedCachedImageFeaturesIndexer(all_features, all_labels, batch_size=batch_size)

            for epoch in tqdm(range(1, n_epochs + 1), desc=f'Epoch'):
                for batch in features_indexer:
                    _ = self.batched_forward_pass(
                        batch['features'],
                        concept,
                        text_labels=batch['labels'],
                        do_backward=True
                    )

                    optimizer.step()
                    optimizer.zero_grad()

    def _load_cache(self, examples: list[ConceptExample], use_concurrency: bool = True, n_jobs: int = 5) -> dict[str,CachedImageFeatures]:
        '''
            Loads all examples into memory to avoid repeated disk I/O.

            Returns:
                A dictionary mapping image feature paths to their corresponding features.
        '''

        logger.debug('Loading examples\' features into memory for caching...')

        if use_concurrency:
            features = Parallel(n_jobs=n_jobs, backend='threading')(delayed(_load_pickle)(ex.image_features_path) for ex in tqdm(examples))
        else:
            features = [_load_pickle(ex.image_features_path) for ex in tqdm(examples)]

        path_to_features = {ex.image_features_path: features for ex, features in zip(examples, features)}

        return path_to_features

    def _load_all_features(
        self,
        dataset: FeatureDataset,
        cache: dict[str,CachedImageFeatures] = None,
        concurrent_load_min_dataset_len: int = 200,
        n_jobs: int = 5
    ) -> CachedImageFeatures:
        '''
            Loads all features for the given dataset into memory.

            If the dataset is large, this function will parallelize the loading of features across multiple processes.

            Returns:
                A batched CachedImageFeatures object with all features and labels loaded.
        '''
        logger.debug('Loading all features for dataset from disk into memory...')

        # Attempt to load features from cache where possible
        inds_to_load = []
        cached_feature_dicts = []

        if cache:
            for index, path in enumerate(dataset.feature_paths):
                if path in cache:
                    feature_dict = dataset.get_metadata(index)
                    feature_dict['features'] = cache[path]

                    cached_feature_dicts.append(feature_dict)
                else:
                    inds_to_load.append(index)
        else:
            inds_to_load = list(range(len(dataset)))

        if inds_to_load:
            if len(dataset) > concurrent_load_min_dataset_len:
                # Using multiprocessing causes an OOM error
                all_feature_dicts = Parallel(n_jobs=n_jobs, backend='threading')(delayed(_load_feature_dict)(dataset, i) for i in tqdm(inds_to_load))
            else:
                all_feature_dicts = [_load_feature_dict(dataset, i) for i in tqdm(inds_to_load)]
        else:
            all_feature_dicts = []

        all_feature_dicts.extend(cached_feature_dicts)

        collate_fn = BatchCachedFeaturesCollate()
        collated = collate_fn(all_feature_dicts)

        return collated['features'], collated['label']

def _load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)

def _load_feature_dict(dataset: FeatureDataset, index: int) -> dict[str, Any]:
    '''
        External to be pickleable
    '''
    return dataset[index]