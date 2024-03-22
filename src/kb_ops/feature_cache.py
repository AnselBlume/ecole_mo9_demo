import os
import torch
from tqdm import tqdm
from model.concept import ConceptKB, Concept
from .feature_pipeline import ConceptKBFeaturePipeline
from image_processing import LocalizeAndSegmentOutput
from model.features import ImageFeatures
from dataclasses import dataclass, field
from typing import Literal
from model.concept import ConceptExample
from PIL.Image import open as open_image
from PIL.Image import Image
import pickle
import logging

logger = logging.getLogger(__name__)

@dataclass
class CachedImageFeatures:
    image_features: torch.Tensor = None # (1, d_img)
    clip_image_features: torch.Tensor = None # (1, d_img)

    region_features: torch.Tensor = None # (n_regions, d_regions)
    clip_region_features: torch.Tensor = None # (n_regions, d_regions)
    region_weights: torch.Tensor = None # (n_regions,); how much to weight each region in all calculations

    trained_attr_img_scores: torch.Tensor = None # (1, n_trained_attrs)
    trained_attr_region_scores: torch.Tensor = None # (n_regions, n_trained_attrs,)

    concept_to_zs_attr_img_scores: dict[str, torch.Tensor] = field(default_factory=dict) # (1, n_zs_attrs)
    concept_to_zs_attr_region_scores: dict[str, torch.Tensor] = field(default_factory=dict) # (n_regions, n_zs_attrs)

    def get_image_features(self, concept_name: str):
        return ImageFeatures(
            image_features=self.image_features,
            clip_image_features=self.clip_image_features,
            region_features=self.region_features,
            clip_region_features=self.clip_region_features,
            region_weights=self.region_weights,
            trained_attr_img_scores=self.trained_attr_img_scores,
            trained_attr_region_scores=self.trained_attr_region_scores,
            zs_attr_img_scores=self.concept_to_zs_attr_img_scores[concept_name],
            zs_attr_region_scores=self.concept_to_zs_attr_region_scores[concept_name]
        )

    def __getitem__(self, concept_name: str):
        return self.get_image_features(concept_name)

    @staticmethod
    def from_image_features(image_features: ImageFeatures):
        cached_features = CachedImageFeatures()

        cached_features.image_features = image_features.image_features
        cached_features.clip_image_features = image_features.clip_image_features
        cached_features.region_features = image_features.region_features
        cached_features.clip_region_features = image_features.clip_region_features
        cached_features.region_weights = image_features.region_weights
        cached_features.trained_attr_img_scores = image_features.trained_attr_img_scores
        cached_features.trained_attr_region_scores = image_features.trained_attr_region_scores

        return cached_features

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
        features_sub_dir = 'features'
    ):
        self.concept_kb = concept_kb
        self.feature_pipeline = feature_pipeline
        self.cache_dir = cache_dir
        self.segmentations_sub_dir = segmentations_sub_dir
        self.features_sub_dir = features_sub_dir

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
    ):
        '''
            If not only_uncached_or_dirty, returns a list of all ConceptExamples in the specified concepts list
            if provided, or all ConceptExamples in the ConceptKB if not. type is then ignored.

            If only_uncached_or_dirty, returns a list of ConceptExamples which are either dirty or do not have
            cached segmentations or features. type specifies whether to check dirty and uncached status for
            segmentations or features.
        '''
        dirty_attr_name = 'are_features_dirty' if type == 'features' else 'are_segmentations_dirty'
        path_attr_name = 'image_features_path' if type == 'features' else 'image_segmentations_path'

        concepts = concepts if concepts else list(self.concept_kb)

        examples = []
        for concept in concepts:
            for example in concept.examples:
                if (
                    not only_uncached_or_dirty
                    or getattr(example, path_attr_name) is None
                    or getattr(example, dirty_attr_name)
                ):
                    examples.append(example)

        return examples

    def _get_segmentation_cache_path(self, example: ConceptExample):
        file_name = os.path.splitext(os.path.basename(example.image_path))[0]
        return f'{self.cache_dir}/{self.segmentations_sub_dir}/{file_name}.pkl'

    def _cache_segmentation(self, example: ConceptExample, **loc_and_seg_kwargs):
        image = self._image_from_example(example)
        segmentations = self.feature_pipeline.get_segmentations(image, **loc_and_seg_kwargs)
        segmentations.input_image_path = example.image_path

        cache_path = self._get_segmentation_cache_path(example)
        with open(cache_path, 'wb') as f:
            pickle.dump(segmentations, f)

        example.image_segmentations_path = cache_path

    def cache_segmentations(self, concepts: list[Concept] = None, only_uncached_or_dirty=True, **loc_and_seg_kwargs):
        '''
            Caches LocalizeAndSegmentOutput pickles to disk for all ConceptExamples in the specified
            concepts list.

            If concepts are not provided, all examples in the ConceptKB will be cached which do not have
            cached segmentations or which are dirty.
        '''
        cache_dir = f'{self.cache_dir}/{self.segmentations_sub_dir}'
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f'Caching segmentations at {cache_dir}')

        examples = self._get_examples(concepts, only_uncached_or_dirty=only_uncached_or_dirty, type='segmentations')

        if not len(examples):
            return

        for example in tqdm(examples):
            self._cache_segmentation(example, **loc_and_seg_kwargs)

    def _get_features_cache_path(self, example: ConceptExample):
        file_name = os.path.splitext(os.path.basename(example.image_path))[0]
        return f'{self.cache_dir}/{self.features_sub_dir}/{file_name}.pkl'

    def cache_features(self, concepts: list[Concept] = None, only_uncached_or_dirty=True):
        '''
            Caches CachedImageFeatures pickles to disk for all ConceptExamples in the specified
            concepts list.

            If concepts are not provided, all examples in the ConceptKB will be cached which do not have
            cached features or which are dirty.
        '''
        cache_dir = f'{self.cache_dir}/{self.features_sub_dir}'
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f'Caching features at {cache_dir}')

        examples = self._get_examples(concepts, only_uncached_or_dirty=only_uncached_or_dirty, type='features')

        if not len(examples):
            return

        for example in tqdm(examples):
            if example.image_segmentations_path is None:
                raise RuntimeError('Segmentations must be cached before features can be cached.')

            # Prepare segmentations
            with open(example.image_segmentations_path, 'rb') as f:
                segmentations = pickle.load(f)

            # Generate zero-shot attributes for each concept
            cached_features = CachedImageFeatures()
            cached_visual_features = None
            cached_clip_visual_features = None
            cached_trained_attr_scores = None

            image = self._image_from_example(example)
            for concept in self.concept_kb:
                # TODO batched feature computation
                feats = self.feature_pipeline.get_features(
                    image,
                    segmentations,
                    [attr.query for attr in concept.zs_attributes],
                    cached_visual_features=cached_visual_features,
                    cached_clip_visual_features=cached_clip_visual_features,
                    cached_trained_attr_scores=cached_trained_attr_scores
                )
                if cached_visual_features is None:
                    cached_visual_features = torch.cat([feats.image_features, feats.region_features], dim=0)

                if cached_clip_visual_features is None:
                    cached_clip_visual_features = torch.cat([feats.clip_image_features, feats.clip_region_features], dim=0)

                if cached_trained_attr_scores is None:
                    cached_trained_attr_scores = torch.cat([feats.trained_attr_img_scores, feats.trained_attr_region_scores], dim=0)

                # Store zero-shot features
                cached_features.concept_to_zs_attr_img_scores[concept.name] = feats.zs_attr_img_scores.cpu()
                cached_features.concept_to_zs_attr_region_scores[concept.name] = feats.zs_attr_region_scores.cpu()
                print(f'Cached zs features for {concept.name} on {example.image_path}')

            # Store non-unique features
            feats.cpu()
            cached_features.image_features = feats.image_features
            cached_features.clip_image_features = feats.clip_image_features
            cached_features.region_features = feats.region_features
            cached_features.clip_region_features = feats.clip_region_features
            cached_features.region_weights = feats.region_weights
            cached_features.trained_attr_img_scores = feats.trained_attr_img_scores
            cached_features.trained_attr_region_scores = feats.trained_attr_region_scores

            # Write to cache
            cache_path = self._get_features_cache_path(example)
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
        # TODO Update when we have separate image features and CLIP features for zs attributes
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
                # Generate zero-shot attribute scores for the new concept
                with open(example.image_segmentations_path, 'rb') as f:
                    segmentations: LocalizeAndSegmentOutput = pickle.load(f)

                # Extract cached features
                dino_device = self.feature_pipeline.feature_extractor.dino_feature_extractor.device
                clip_device = self.feature_pipeline.feature_extractor.clip_feature_extractor.device

                visual_features = torch.cat([cached_features.image_features, cached_features.region_features], dim=0).to(dino_device)
                clip_visual_features = torch.cat([cached_features.clip_image_features, cached_features.clip_region_features], dim=0).to(clip_device)
                trained_attr_scores = torch.cat([cached_features.trained_attr_img_scores, cached_features.trained_attr_region_scores], dim=0).to(clip_device)

                feats = self.feature_pipeline.get_features(
                    None, # Don't need to pass image since passing visual features
                    segmentations,
                    [attr.query for attr in concept.zs_attributes],
                    cached_visual_features=visual_features,
                    cached_clip_visual_features=clip_visual_features,
                    cached_trained_attr_scores=trained_attr_scores
                )

                cached_features.concept_to_zs_attr_img_scores[concept.name] = feats.zs_attr_img_scores.cpu()
                cached_features.concept_to_zs_attr_region_scores[concept.name] = feats.zs_attr_region_scores.cpu()

                # Update cached features
                with open(example.image_features_path, 'wb') as f:
                    pickle.dump(cached_features, f)

                logger.debug(f'Added concept {concept.name} to cached features for example {example.image_path}')