from .base import ConceptKBTrainerBase
import torch
from model.concept import Concept
from kb_ops.dataset import FeatureDataset
from joblib import Parallel, delayed
from typing import Any
from tqdm import tqdm
from kb_ops.dataset import BatchCachedFeaturesCollate, BatchedCachedImageFeaturesIndexer
from kb_ops.caching import CachedImageFeatures
from .outputs import TrainOutput, ValidationOutput
import logging

logger = logging.getLogger(__file__)

class ConceptKBInMemoryTrainerMixin(ConceptKBTrainerBase):
    def train_in_memory(
        self,
        train_ds: FeatureDataset,
        n_epochs: int,
        lr: float,
        concepts: list[Concept] = None,
        batch_size: int = 64
    ):
        '''
            Trains the concept predictors using the given dataset one-by-one for the specified number of epochs.
            Identical to train_minimal, but loads all features into memory before training to avoid repeated disk I/O.
        '''
        concept_to_train_dataset = self._concepts_to_datasets(train_ds, concepts=concepts)
        self.concept_kb.train()

        prog_bar = tqdm(concept_to_train_dataset.items())
        for concept, dataset in prog_bar:
            prog_bar.set_description(f'Training concept "{concept.name}"')

            optimizer = torch.optim.Adam(concept.predictor.parameters(), lr=lr)
            all_features, all_labels = self._load_all_features(dataset)
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

    def _load_all_features(self, dataset: FeatureDataset, concurrent_load_min_dataset_len: int = 200, n_jobs: int = 5) -> CachedImageFeatures:
        '''
            Loads all features for the given dataset into memory.

            If the dataset is large, this function will parallelize the loading of features across multiple processes.

            Returns:
                A batched CachedImageFeatures object with all features and labels loaded.
        '''
        logger.debug('Loading all features for dataset from disk into memory...')


        if len(dataset) > concurrent_load_min_dataset_len:
            # Using multiprocessing causes an OOM error
            all_feature_dicts = Parallel(n_jobs=n_jobs, backend='threading')(delayed(_load_feature_dict)(dataset, i) for i in tqdm(range(len(dataset))))
        else:
            all_feature_dicts = [_load_feature_dict(dataset, i) for i in tqdm(range(len(dataset)))]

        collate_fn = BatchCachedFeaturesCollate()
        collated = collate_fn(all_feature_dicts)

        return collated['features'], collated['label']

def _load_feature_dict(dataset: FeatureDataset, index: int) -> dict[str, Any]:
    '''
        External to be pickleable
    '''
    return dataset[index]