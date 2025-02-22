from .base import ConceptKBTrainerBase
import numpy as np
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

    def train_batched_in_memory(
        self,
        train_ds: FeatureDataset,
        val_ds: Optional[FeatureDataset],
        n_epochs: int,
        lr: float,
        concepts: list[Concept] = None,
        batch_size: int = 64,
        ckpt_every_n_epochs: int = 1,
        ckpt_dir: str = 'checkpoints',
        ckpt_fmt: str = 'concept_kb_epoch_{epoch}.pt',
        cache_full_dataset: bool = True
    ) -> TrainOutput:

        # For each global concept, create a dataset with its examples
        concept_to_train_dataset = self._concepts_to_datasets(train_ds, concepts=concepts)
        concept_to_val_dataset = self._concepts_to_datasets(val_ds, concepts=concepts) if val_ds else None
        concepts = list(concept_to_train_dataset.keys())

        train_outputs = []
        val_outputs: list[ValidationOutput] = []

        concept_to_optimizer = {c.name : torch.optim.Adam(c.predictor.parameters(), lr=lr) for c in concepts}

        # concept_to_train_dl = self._concepts_to_dataloaders(concept_to_train_dataset, is_train=True, batch_size=batch_size, **dataloader_kwargs)

        # Load cache
        examples_to_cache = list(self.concept_kb.global_negatives) # Copy to avoid modifying global_negatives

        if cache_full_dataset:
            concept_examples = list(chain.from_iterable([c.examples for c in self.concept_kb]))
            examples_to_cache.extend(concept_examples)

        cache: dict[str,CachedImageFeatures] = self._load_cache(examples_to_cache)

        for epoch in range(1, n_epochs + 1):
            logger.info(f'======== Starting Epoch {epoch}/{n_epochs} ========')
            self.concept_kb.train()

            concepts_outputs = {}
            for concept, dataset in tqdm(concept_to_train_dataset.items(), desc=f'Epoch {epoch}/{n_epochs}'):
                optimizer = concept_to_optimizer[concept.name]
                all_features, all_labels = self._load_all_features(dataset, cache=cache)
                features_indexer = BatchedCachedImageFeaturesIndexer(all_features, all_labels, batch_size=batch_size)

                # Train concept predictor
                for batch in features_indexer:
                    outputs = self.batched_forward_pass(
                        batch['features'],
                        concept,
                        text_labels=batch['labels'],
                        do_backward=True
                    )
                    concepts_outputs.setdefault(concept.name, []).append(outputs)

                    optimizer.step()
                    optimizer.zero_grad()

            train_outputs.append(concepts_outputs)

            # Compute loss for logging
            concepts_to_losses = {
                concept_name : sum(output.loss for output in outputs) / len(outputs)
                for concept_name, outputs in concepts_outputs.items()
            }
            train_loss = sum(concepts_to_losses.values()) / len(concepts_to_losses)

            self.log({'train_loss': train_loss, 'epoch': epoch})

            if epoch % ckpt_every_n_epochs == 0 and ckpt_dir:
                ckpt_path = self._get_ckpt_path(ckpt_dir, ckpt_fmt, epoch)
                self.concept_kb.save(ckpt_path)
                logger.info(f'Saved checkpoint at {ckpt_path}')

            # Validate
            if concept_to_val_dataset:
                # TODO batched validation using concept_to_val_dataset
                # For now, just do single example validation
                val_dl = self._get_dataloader(val_ds, is_train=False)
                outputs = self.validate(val_dl)
                val_outputs.append(outputs)

                self.log({
                    'val_loss': outputs.loss,
                    'val_component_acc': outputs.component_accuracy,
                    'val_non_component_acc': outputs.non_component_accuracy,
                    'epoch': epoch
                })

                logger.info(
                    f'Validation loss: {outputs.loss},'
                    + f' Validation component accuracy: {outputs.component_accuracy:.4f},'
                    + f' Validation non-component accuracy: {outputs.non_component_accuracy:.4f}'
                )

        # Construct return dictionary
        val_losses = [output.loss for output in val_outputs] if val_ds else None
        best_ckpt_epoch = np.argmin(val_losses) if val_losses else None
        best_ckpt_path = self._get_ckpt_path(ckpt_dir, ckpt_fmt, best_ckpt_epoch) if val_losses else None

        train_output = TrainOutput(
            best_ckpt_epoch=best_ckpt_epoch,
            best_ckpt_path=best_ckpt_path,
            train_outputs=train_outputs,
            val_outputs=val_outputs
        )

        return train_output

def _load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)

def _load_feature_dict(dataset: FeatureDataset, index: int) -> dict[str, Any]:
    '''
        External to be pickleable
    '''
    return dataset[index]