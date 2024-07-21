import hashlib
import logging
import os
import pickle
from typing import Literal

import torch
from filelock import FileLock
from kb_ops.feature_pipeline import ConceptKBFeaturePipeline
from model.concept import Concept, ConceptExample, ConceptKB
from PIL.Image import Image
from PIL.Image import open as open_image
from torch import Tensor
from tqdm import tqdm

from .cached_image_features import CachedImageFeatures

logger = logging.getLogger(__name__)

FILE_LOCK_TIMEOUT_S = 10
LOCK_DIR = "/shared/nas2/knguye71/ecole-june-demo/lock_dir/"

class ConceptKBFeatureCacher:
    '''
        Assumes that every ConceptExample has an image_path.
    '''
    def __init__(
        self,
        concept_kb: ConceptKB,
        feature_pipeline: ConceptKBFeaturePipeline,
        cache_dir = 'feature_cache',
        segmentations_sub_dir = 'segmentations',
        features_sub_dir = 'features',
        infer_localize_from_component: bool = True
    ):
        '''
            infer_localize_from_component: If True, will infer whether to localize based on whether the concept
                is a component concept or not, localizing only if it is not a component concept.
                If False, localization will default to the LocalizerAndSegmenter's Config's default do_localize value.
        '''
        self.concept_kb = concept_kb
        self.feature_pipeline = feature_pipeline
        self.cache_dir = cache_dir
        self.segmentations_sub_dir = segmentations_sub_dir
        self.features_sub_dir = features_sub_dir
        self.infer_localize_from_component = infer_localize_from_component

    @property
    def segmentations_dir(self):
        return f'{self.cache_dir}/{self.segmentations_sub_dir}'

    @property
    def features_dir(self):
        return f'{self.cache_dir}/{self.features_sub_dir}'

    def _image_from_example(self, example: ConceptExample) -> Image:
        if example.image is not None:
            return example.image

        return open_image(example.image_path).convert('RGB')

    def _get_examples(
        self,
        concepts: list[Concept] = None,
        only_uncached_or_dirty: bool = False,
        type: Literal['segmentations', 'features'] = None,
        include_global_negatives: bool = True
    ):
        '''
            If not only_uncached_or_dirty, returns a list of all ConceptExamples in the specified concepts list
            if provided, or all ConceptExamples in the ConceptKB if not. type is then ignored.

            If only_uncached_or_dirty, returns a list of ConceptExamples which are either dirty or do not have
            cached segmentations or features. type specifies whether to check dirty and uncached status for
            segmentations or features.
        '''
        if only_uncached_or_dirty and type not in ['segmentations', 'features']:
            raise ValueError(f'only_uncached_or_dirty is True; type {type} must be one of {{"segmentations", "features"}}')

        dirty_attr_name = 'are_features_dirty' if type == 'features' else 'are_segmentations_dirty'
        path_attr_name = 'image_features_path' if type == 'features' else 'image_segmentations_path'

        concepts = concepts if concepts else list(self.concept_kb)

        def include_example_filter(example):
            return (
                not only_uncached_or_dirty
                or getattr(example, path_attr_name) is None
                or getattr(example, dirty_attr_name)
            )

        # Each concept's examples
        examples = []
        for concept in concepts:
            for example in concept.examples:
                if include_example_filter(example):
                    examples.append(example)

        # Global negative examples
        if include_global_negatives:
            for example in self.concept_kb.global_negatives:
                if include_example_filter(example):
                    examples.append(example)

        return examples

    def _hash_str(self, s: str) -> str:
        md5 = hashlib.md5()
        md5.update(s.encode())
        hex = md5.hexdigest()

        return hex

    def _get_segmentation_cache_path(self, example: ConceptExample):
        return os.path.realpath(f'{self.cache_dir}/{self.segmentations_sub_dir}/{self._hash_str(example.image_path)}.pkl')

    def _cache_segmentation(self, example: ConceptExample, prog_bar: tqdm = None, **loc_and_seg_kwargs):
        image = self._image_from_example(example)

        if self.infer_localize_from_component:
            # Don't perform localization if it is a component concept. Otherwise, use the LocalizerAndSegmenter's default
            # If the concept has containing concepts, it is a component
            if example.concept_name and self.concept_kb[example.concept_name].containing_concepts:
                loc_and_seg_kwargs['do_localize'] = False

        segmentations = self.feature_pipeline.get_segmentations(image, **loc_and_seg_kwargs).cpu()
        segmentations.input_image_path = example.image_path

        cache_path = self._get_segmentation_cache_path(example)

        if prog_bar is not None:
            prog_bar.set_description(f'Caching segmentations {os.path.basename(example.image_path)}')

        lock_path = LOCK_DIR + os.path.basename(cache_path) + ".lock"

        with FileLock(lock_path, timeout=FILE_LOCK_TIMEOUT_S):
            with open(cache_path, 'wb') as f:
                pickle.dump(segmentations, f)

        example.image_segmentations_path = cache_path

    def cache_segmentations(self, concepts: list[Concept] = None, only_uncached_or_dirty=True, include_global_negatives: bool = True, **loc_and_seg_kwargs):
        '''
            Caches LocalizeAndSegmentOutput pickles to disk for all ConceptExamples in the specified
            concepts list.

            If concepts are not provided, all examples in the ConceptKB will be cached which do not have
            cached segmentations or which are dirty.
        '''
        cache_dir = f'{self.cache_dir}/{self.segmentations_sub_dir}'
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f'Caching segmentations at {cache_dir}')

        examples = self._get_examples(concepts, only_uncached_or_dirty=only_uncached_or_dirty, type='segmentations', include_global_negatives=include_global_negatives)

        if not len(examples):
            return

        prog_bar = tqdm(examples)
        for example in prog_bar:
            self._cache_segmentation(example, prog_bar=prog_bar, **loc_and_seg_kwargs)

    def _get_features_cache_path(self, example: ConceptExample):
        return f'{self.cache_dir}/{self.features_sub_dir}/{self._hash_str(example.image_path)}.pkl'

    def cache_features(self, concepts: list[Concept] = None, only_uncached_or_dirty=True, include_global_negatives: bool = True):
        '''
            Caches CachedImageFeatures pickles to disk for all ConceptExamples in the specified
            concepts list.

            If concepts are not provided, all examples in the ConceptKB will be cached which do not have
            cached features or which are dirty.
        '''
        cache_dir = f'{self.cache_dir}/{self.features_sub_dir}'
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f'Caching features at {cache_dir}')

        examples = self._get_examples(concepts, only_uncached_or_dirty=only_uncached_or_dirty, type='features', include_global_negatives=include_global_negatives)
        print("Caching features for", len(examples), "examples")

        if not len(examples):
            return

        prog_bar = tqdm(examples)
        for example in prog_bar:
            if example.image_segmentations_path is None:
                raise RuntimeError('Segmentations must be cached before features can be cached.')

            lock_path = LOCK_DIR + os.path.basename(example.image_segmentations_path) + '.lock'

            with FileLock(lock_path, timeout=FILE_LOCK_TIMEOUT_S):
                # Prepare segmentations
                with open(example.image_segmentations_path, 'rb') as f:
                    segmentations = pickle.load(f)

            # Generate zero-shot attributes for each concept
            cached_features = None

            image = self._image_from_example(example)
            for concept in self.concept_kb:
                # TODO batched feature computation
                feats = self.feature_pipeline.get_concept_predictor_features(image, segmentations, concept, cached_features=cached_features)

                if cached_features is None:
                    cached_features = CachedImageFeatures.from_image_features(feats)

                # Store concept-specific features
                cached_features.update_concept_predictor_features(concept, feats, store_component_concept_scores=self.feature_pipeline.config.compute_component_concept_scores)

            cached_features.cpu()

            # Write to cache
            cache_path = self._get_features_cache_path(example)

            prog_bar.set_description(f'Caching features {os.path.basename(example.image_path)}')

            lock_path = LOCK_DIR + os.path.basename(cache_path) + '.lock'
            with FileLock(lock_path, timeout=FILE_LOCK_TIMEOUT_S):
                with open(cache_path, 'wb') as f:
                    pickle.dump(cached_features, f)

            example.image_features_path = cache_path

    def recache_zs_attr_features(
        self,
        concept: Concept,
        examples: list[ConceptExample] = None,
        only_not_present: bool = False,
        batch_size: int = 128
    ):
        '''
            Recaches zero-shot attribute features for the specified Concept across all Concepts' examples in the
            ConceptKB.

            If examples are provided, only recaches features for the specified examples (to save time instead of
            recaching for all in the concept_kb).

            If only_not_present is True, only recaches features for examples which do not have the specified
            concept's zero-shot attribute features. So this will not overwrite existing zs attribute features.
        '''
        if not len(concept.zs_attributes):
            return

        examples = examples if examples else self._get_examples([concept])

        example_batch: list[ConceptExample] = []
        visual_features_batch: list[Tensor] = []
        n_features_per_example = []

        # Precompute zs attribute features
        clip_device = self.feature_pipeline.feature_extractor.zs_attr_predictor.device

        with torch.no_grad():
            zs_attr_features = self.feature_pipeline.feature_extractor.clip_feature_extractor(
                texts=[attr.query for attr in concept.zs_attributes]
            ).to(clip_device)

        for i, example in enumerate(examples):
            if example.image_features_path is None:
                continue
            else:
                lock_path = LOCK_DIR + os.path.basename(example.image_features_path) + '.lock'
                with FileLock(lock_path, timeout=FILE_LOCK_TIMEOUT_S):
                    with open(example.image_features_path, 'rb') as f:
                        cached_features: CachedImageFeatures = pickle.load(f)

                if ( # Skip if only recaching scores where they are not present, and it is already present in this example
                    only_not_present
                    and (
                        concept.name in cached_features.concept_to_zs_attr_img_scores
                        or concept.name in cached_features.concept_to_zs_attr_region_scores
                    )
                ):
                    continue

                # Add to batch
                example_visual_features = torch.cat([
                    cached_features.clip_image_features,
                    cached_features.clip_region_features,
                ], dim=-2) # (n_regions + 1, d)

                example_batch.append(example)
                visual_features_batch.append(example_visual_features)
                n_features_per_example.append(example_visual_features.shape[0])

                # Compute batch
                if len(example_batch) == batch_size or i == len(examples) - 1:
                    batch_zs_scores = self.feature_pipeline.feature_extractor.get_zero_shot_attr_scores(
                        torch.cat(visual_features_batch, dim=-2).to(clip_device),
                        zs_attr_features=zs_attr_features
                    ).cpu() # (sum_i (1 + n_regions_i), n_zs_attrs)

                    # Split scores back into individual examples
                    offset = 0
                    for example, n_features in zip(example_batch, n_features_per_example):
                        zs_attr_scores = batch_zs_scores[offset:offset+n_features]
                        zs_attr_img_scores = zs_attr_scores[0:1] # (1, n_zs_attrs)
                        zs_attr_region_scores = zs_attr_scores[1:] # (n_regions, n_zs_attrs)

                        offset += n_features
                        lock_path = LOCK_DIR + os.path.basename(example.image_features_path) + '.lock'
                        with FileLock(lock_path, timeout=FILE_LOCK_TIMEOUT_S):
                            # Update cached features
                            with open(example.image_features_path, 'r+b') as f:
                                cached_features: CachedImageFeatures = pickle.load(f)
                                cached_features.concept_to_zs_attr_img_scores[concept.name] = zs_attr_img_scores
                                cached_features.concept_to_zs_attr_region_scores[concept.name] = zs_attr_region_scores

                                f.seek(0) # Go back to start of file
                                pickle.dump(cached_features, f)
                                f.truncate() # Truncate to new size

                    example_batch.clear()
                    visual_features_batch.clear()
                    n_features_per_example.clear()

    def recache_component_concept_scores(self, concept: Concept, examples: list[ConceptExample] = None):
        '''
            Recaches component concept scores for the specified Concept across all Concepts' examples in the
            ConceptKB. Automatically computes scores only for concepts which are not already cached (handled in
            call to ConceptKBFeaturePipeline._get_component_concept_scores).

            NOTE Should be called ONLY if the component concept scores are fixed for a given concept-image pair
            (e.g. when using a fixed model, like DesCo, to compute the scores).

            If examples are provided, only recaches features for the specified examples (to save time instead of
            recaching for all in the concept_kb).
        '''
        # TODO Batch this process
        if not self.feature_pipeline.config.compute_component_concept_scores:
            raise RuntimeError('Component concept scores are not being computed by the feature pipeline.')

        examples = examples if examples else self._get_examples([concept])

        for example in examples:
            if example.image_features_path is None:
                continue

            lock_path = LOCK_DIR + os.path.basename(example.image_features_path) + '.lock'
            with FileLock(lock_path, timeout=FILE_LOCK_TIMEOUT_S):
                with open(example.image_features_path, 'rb') as f:
                    cached_features: CachedImageFeatures = pickle.load(f)

            # Need to update the cache if the score for any component concept of this concept is missing from the cache
            if any(cached_features.component_concept_scores.get(name, None) is None for name in concept.component_concepts):
                # While this looks like it is recomputing all component concept scores for the concept, the _get_component_concept_scores
                # method actually only computes the scores for those which don't already exist
                cached_features.cuda()
                component_scores = self.feature_pipeline._get_component_concept_scores(concept, cached_features).cpu()

                for component_concept, component_score in zip(concept.component_concepts, component_scores):
                    cached_features.component_concept_scores[component_concept] = component_score

                cached_features.cpu() # Back to CPU for saving
                lock_path = LOCK_DIR + os.path.basename(example.image_features_path) + '.lock'

                with FileLock(lock_path, timeout=FILE_LOCK_TIMEOUT_S):
                    # Update cached features
                    with open(example.image_features_path, 'wb') as f:
                        pickle.dump(cached_features, f)

                logger.debug(f'Added concept {concept.name}\'s component concept scores to cached features for example {example.image_path}')
