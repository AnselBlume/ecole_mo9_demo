import os
import sys
from tqdm import tqdm
from model.concept import ConceptKB, Concept
from ..feature_pipeline import ConceptKBFeaturePipeline
from typing import Literal
from model.concept import ConceptExample
from PIL.Image import open as open_image
from PIL.Image import Image
import pickle
import logging
from .cached_image_features import CachedImageFeatures
import hashlib

logger = logging.getLogger(__name__)

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
        return f'{self.cache_dir}/{self.segmentations_sub_dir}/{self._hash_str(example.image_path)}.pkl'

    def _cache_segmentation(self, example: ConceptExample, prog_bar: tqdm = None, **loc_and_seg_kwargs):
        image = self._image_from_example(example)

        if self.infer_localize_from_component:
            # Perform localization only if the concept is not a component concept
            if example.concept_name and self.concept_kb[example.concept_name].containing_concepts:
                loc_and_seg_kwargs['do_localize'] = True

        segmentations = self.feature_pipeline.get_segmentations(image, **loc_and_seg_kwargs)
        segmentations.input_image_path = example.image_path

        cache_path = self._get_segmentation_cache_path(example)

        if prog_bar is not None:
            prog_bar.set_description(f'Caching segmentations {os.path.basename(example.image_path)}')

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

            with open(cache_path, 'wb') as f:
                pickle.dump(cached_features, f)

            example.image_features_path = cache_path

    def recache_zs_attr_features(self, concept: Concept, examples: list[ConceptExample] = None, only_not_present: bool = False):
        '''
            Recaches zero-shot attribute features for the specified Concept across all Concepts' examples in the
            ConceptKB.

            If examples are provided, only recaches features for the specified examples (to save time instead of
            recaching for all in the concept_kb).

            If only_not_present is True, only recaches features for examples which do not have the specified
            concept's zero-shot attribute features. So this will not overwrite existing zs attribute features.
        '''
        examples = examples if examples else self._get_examples([concept])

        for example in examples:
            if example.image_features_path is None:
                continue

            with open(example.image_features_path, 'rb') as f:
                cached_features: CachedImageFeatures = pickle.load(f)

            if (
                not only_not_present
                or concept.name not in cached_features.concept_to_zs_attr_img_scores
                or concept.name not in cached_features.concept_to_zs_attr_region_scores
            ):
                cached_features.cuda()

                zs_attr_img_scores, zs_attr_region_scores = self.feature_pipeline._get_zero_shot_attr_scores(concept, cached_features, recompute_scores=True)

                cached_features.concept_to_zs_attr_img_scores[concept.name] = zs_attr_img_scores
                cached_features.concept_to_zs_attr_region_scores[concept.name] = zs_attr_region_scores

                cached_features.cpu() # Back to CPU for saving

                # Update cached features
                with open(example.image_features_path, 'wb') as f:
                    pickle.dump(cached_features, f)

                logger.debug(f'Added concept {concept.name} zero-shot attributes to cached features for example {example.image_path}')

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
        if not self.feature_pipeline.config.compute_component_concept_scores:
            raise RuntimeError('Component concept scores are not being computed by the feature pipeline.')

        examples = examples if examples else self._get_examples([concept])

        for example in examples:
            if example.image_features_path is None:
                continue

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

                # Update cached features
                with open(example.image_features_path, 'wb') as f:
                    pickle.dump(cached_features, f)

                logger.debug(f'Added concept {concept.name}\'s component concept scores to cached features for example {example.image_path}')