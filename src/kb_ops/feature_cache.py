import os
import torch
from tqdm import tqdm
from model.concept import ConceptKB, Concept
from .feature_pipeline import ConceptKBFeaturePipeline
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

    region_features: torch.Tensor = None # (n_regions, d_regions)
    region_weights: torch.Tensor = None # (n_regions,); how much to weight each region in all calculations

    trained_attr_img_scores: torch.Tensor = None # (1, n_trained_attrs)
    trained_attr_region_scores: torch.Tensor = None # (n_regions, n_trained_attrs,)

    concept_to_zs_attr_img_scores: dict[str, torch.Tensor] = field(default_factory=dict) # (1, n_zs_attrs)
    concept_to_zs_attr_region_scores: dict[str, torch.Tensor] = field(default_factory=dict) # (n_regions, n_zs_attrs)

    def get_image_features(self, concept_name: str):
        return ImageFeatures(
            image_features=self.image_features,
            region_features=self.region_features,
            region_weights=self.region_weights,
            trained_attr_img_scores=self.trained_attr_img_scores,
            trained_attr_region_scores=self.trained_attr_region_scores,
            zs_attr_img_scores=self.concept_to_zs_attr_img_scores[concept_name],
            zs_attr_region_scores=self.concept_to_zs_attr_region_scores[concept_name]
        )

    def __getitem__(self, concept_name: str):
        return self.get_image_features(concept_name)

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

    def _image_from_example(self, example: ConceptExample) -> Image:
        if example.image is not None:
            return example.image

        return open_image(example.image_path).convert('RGB')

    def _get_uncached_or_dirty_examples(self, type: Literal['segmentations', 'features']):
        dirty_attr_name = 'are_features_dirty' if type == 'features' else 'are_segmentations_dirty'
        path_attr_name = 'image_features_path' if type == 'features' else 'image_segmentations_path'

        return [
            example for concept in self.concept_kb
            for example in concept.examples
            if getattr(example, path_attr_name) is None or getattr(example, dirty_attr_name)
        ]

    def _examples_from_concepts(self, concepts: list[Concept]):
        return [
            example for concept in concepts
            for example in concept.examples
        ]

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

    def cache_segmentations(self, concepts: list[Concept] = None, **loc_and_seg_kwargs):
        '''
            Caches LocalizeAndSegmentOutput pickles to disk for all ConceptExamples in the specified
            concepts list.

            If concepts are not provided, all examples in the ConceptKB will be cached which do not have
            cached segmentations or which are dirty.
        '''
        cache_dir = f'{self.cache_dir}/{self.segmentations_sub_dir}'
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f'Caching segmentations at {cache_dir}')

        examples = self._get_uncached_or_dirty_examples('segmentations') if concepts is None else self._examples_from_concepts(concepts)
        for example in tqdm(examples):
            self._cache_segmentation(example, **loc_and_seg_kwargs)

    def _get_features_cache_path(self, example: ConceptExample):
        file_name = os.path.splitext(os.path.basename(example.image_path))[0]
        return f'{self.cache_dir}/{self.features_sub_dir}/{file_name}.pkl'

    def cache_features(self, concepts: list[Concept] = None):
        '''
            Caches CachedImageFeatures pickles to disk for all ConceptExamples in the specified
            concepts list.

            If concepts are not provided, all examples in the ConceptKB will be cached which do not have
            cached features or which are dirty.
        '''
        cache_dir = f'{self.cache_dir}/{self.features_sub_dir}'
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f'Caching features at {cache_dir}')

        examples = self._get_uncached_or_dirty_examples('features') if concepts is None else self._examples_from_concepts(concepts)
        for example in tqdm(examples):
            if example.image_segmentations_path is None:
                raise RuntimeError('Segmentations must be cached before features can be cached.')

            # Prepare segmentations
            with open(example.image_segmentations_path, 'rb') as f:
                segmentations = pickle.load(f)

            # Generate zero-shot attributes for each concept
            cached_features = CachedImageFeatures()
            cached_visual_features = None
            cached_trained_attr_scores = None

            for concept in self.concept_kb:
                # TODO batched feature computation
                feats = self.feature_pipeline.get_features(
                    self._image_from_example(example),
                    segmentations,
                    [attr.query for attr in concept.zs_attributes],
                    cached_visual_features=cached_visual_features,
                    cached_trained_attr_scores=cached_trained_attr_scores
                )
                if cached_visual_features is None:
                    cached_visual_features = torch.cat([feats.image_features, feats.region_features], dim=0)

                if cached_trained_attr_scores is None:
                    cached_trained_attr_scores = torch.cat([feats.trained_attr_img_scores, feats.trained_attr_region_scores], dim=0)

                # Store zero-shot features
                cached_features.concept_to_zs_attr_img_scores[concept.name] = feats.zs_attr_img_scores.cpu()
                cached_features.concept_to_zs_attr_region_scores[concept.name] = feats.zs_attr_region_scores.cpu()


            # Store non-unique features
            feats.cpu()
            cached_features.image_features = feats.image_features
            cached_features.region_features = feats.region_features
            cached_features.region_weights = feats.region_weights
            cached_features.trained_attr_img_scores = feats.trained_attr_img_scores
            cached_features.trained_attr_region_scores = feats.trained_attr_region_scores

            # Write to cache
            cache_path = self._get_features_cache_path(example)
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_features, f)

            example.image_features_path = cache_path