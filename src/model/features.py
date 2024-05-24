from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional
from model.concept import Concept
import logging

logger = logging.getLogger(__name__)

import torch
from dataclasses import dataclass

@dataclass
class ImageFeatures:
    #####################################
    # Features for internal calculation #
    #####################################
    image_features: torch.Tensor = None # (1, d_img)
    clip_image_features: torch.Tensor = None # (1, d_img)

    region_features: torch.Tensor = None # (n_regions, d_regions)
    clip_region_features: torch.Tensor = None # (n_regions, d_regions)
    region_weights: torch.Tensor = None # (n_regions,); how much to weight each region in all calculations

    trained_attr_img_scores: torch.Tensor = None # (1, n_trained_attrs)
    trained_attr_region_scores: torch.Tensor = None # (n_regions, n_trained_attrs,)

    #############################################
    # Features computed via batched calculation #
    #############################################
    # Tensor of shape (1 + 1 + 2*n_learned_attrs + 2*n_zs_attrs,) where the first and second elmts are
    # the image and region scores, respectively
    all_scores: torch.Tensor = None

    @classmethod
    def from_image_features(cls, image_features: ImageFeatures):
        features = cls()

        features.image_features = image_features.image_features
        features.clip_image_features = image_features.clip_image_features
        features.region_features = image_features.region_features
        features.clip_region_features = image_features.clip_region_features
        features.region_weights = image_features.region_weights
        features.trained_attr_img_scores = image_features.trained_attr_img_scores
        features.trained_attr_region_scores = image_features.trained_attr_region_scores

        return features

    def to(self, device):
        '''
            Shifts all tensors to the specified device.
        '''
        for field in self.__dataclass_fields__:
            attr = getattr(self, field)

            if isinstance(attr, torch.Tensor):
                setattr(self, field, attr.to(device))

        return self

    def cpu(self):
        '''
            Moves all tensors to the CPU.
        '''
        return self.to('cpu')

    def cuda(self):
        '''
            Moves all tensors to the GPU.
        '''
        return self.to('cuda')

@dataclass
class ConceptPredictorFeatures(ImageFeatures):
    zs_attr_img_scores: torch.Tensor = None # (1, n_zs_attrs)
    zs_attr_region_scores: torch.Tensor = None # (n_regions, n_zs_attrs)

    component_concept_scores: torch.Tensor = None # (1, n_component_concepts,)

@dataclass
class CachedImageFeatures(ImageFeatures):
    concept_to_zs_attr_img_scores: dict[str, torch.Tensor] = field(default_factory=dict) # (1, n_zs_attrs)
    concept_to_zs_attr_region_scores: dict[str, torch.Tensor] = field(default_factory=dict) # (n_regions, n_zs_attrs)

    component_concept_scores: dict[str, torch.Tensor] = field(
        default_factory=dict,
        metadata={'help': 'Stores component concept scores for each concept in the ConceptKB.'
                        + ' Only stored if static for a given image, regardless of how ConceptPredictors change (i.e. if using DesCo to detect).'}
    ) # (,)

    def get_concept_predictor_features(self, concept_name: str):
        return ConceptPredictorFeatures(
            image_features=self.image_features,
            clip_image_features=self.clip_image_features,
            region_features=self.region_features,
            clip_region_features=self.clip_region_features,
            region_weights=self.region_weights,
            trained_attr_img_scores=self.trained_attr_img_scores,
            trained_attr_region_scores=self.trained_attr_region_scores,
            zs_attr_img_scores=self.concept_to_zs_attr_img_scores[concept_name],
            zs_attr_region_scores=self.concept_to_zs_attr_region_scores[concept_name],
            component_concept_scores=self.component_concept_scores.get(concept_name, None)
        )

    def __getitem__(self, concept_name: str):
        return self.get_concept_predictor_features(concept_name)

    def update_concept_predictor_features(self, concept: Concept, features: ConceptPredictorFeatures, store_component_concept_scores: bool = True):
        self.concept_to_zs_attr_img_scores[concept.name] = features.zs_attr_img_scores.cpu()
        self.concept_to_zs_attr_region_scores[concept.name] = features.zs_attr_region_scores.cpu()

        if store_component_concept_scores:
            component_concept_scores = features.component_concept_scores.cpu()
            assert len(concept.component_concepts) == len(component_concept_scores)

            # Overwriting existing scores is okay because the score is fixed for a given image-concept pair
            for component_concept_name, score in zip(concept.component_concepts, component_concept_scores):
                self.component_concept_scores[component_concept_name] = score

        return self

@dataclass
class FeatureMetadata:
    name: str = field(default=None, metadata={'help': 'Name of the feature.'})
    query: str = field(default=None, metadata={'help': 'Key in the feature vector.'})
    is_necessary: bool = field(default=False, metadata={'help': 'Whether the feature is necessary instead of descriptive.'})

class FeatureGroup(nn.Module):
    def __init__(
        self,
        *,
        indices: list[int] = None,
        range: tuple[int,int] = None,
        use_equal_weights: bool = True,
        weight_by_probs: bool = False,
        copy_input_features: bool = True,
        feature_metadata: list[Optional[FeatureMetadata]] = []
    ):
        '''
            indices (Optional[list[int]]): Indicates the specific indices to use from the input feature vector.
            range (Optional[tuple[int,int]]) Indicates the range of indices to use from the input feature vector (end index exclusive).
            use_equal_weights (bool): Whether to use a single weight for all features in the group.
            weight_by_probs (bool): Whether to normalize learned weights into probabilities before multiplying.
            copy_input_features (bool): Whether to copy the input tensor before modifying it (to avoid modifying the original tensor in-place).
            feature_metadata (Optional[list[FeatureMetadata]]): Metadata for each feature in the group.
        '''
        super().__init__()

        assert (range is None) ^ (indices is None), 'Exactly one of range or indices must be provided.'

        self.start_ind, self.end_ind = range if range else (None, None)
        self.indices = indices

        # Can use self.n_features property from this point onwards

        self.weight_by_probs = weight_by_probs
        self.copy_input_features = copy_input_features

        if use_equal_weights:
            if weight_by_probs:
                logger.warning('Weighting by probabilities with equal weights; this will not be trainable and will use uniform distribution')

            else:
                self.feature_weights = nn.Parameter(torch.tensor([1 / self.n_features]))

        else:
            self.feature_weights = nn.Parameter(torch.ones(self.n_features) / self.n_features)

        if not feature_metadata:
            feature_metadata = [None for _ in range(self.n_features)]

    @property
    def n_features(self):
        return len(self.indices) if self.indices else (self.end_ind - self.start_ind)

    def _convert_to_index_based(self):
        self.indices = list(range(self.start_ind, self.end_ind))
        self.start_ind, self.end_ind = None, None

    def add_index(self, index: int, weight: float = None, metadata: FeatureMetadata = None):
        if not self.indices: # Range-based
            logger.info('Feature group is range-based; converting to index-based.')
            self._convert_to_index_based()

        self.indices.append(index)
        self.feature_metadata.append(metadata)

        if not self.use_equal_weights:
            if weight is None:
                weight = self.feature_weights.mean().item()

            self.feature_weights = nn.Parameter(
                torch.cat([
                    self.feature_weights,
                    torch.tensor([weight], device=self.feature_weights.device)
                ])
            )

    def remove_index(self, index: int):
        if not self.indices:
            logger.info('Feature group is range-based; converting to index-based.')
            self._convert_to_index_based()

        removal_ind = self.indices.index(index)
        self.indices = self.indices[:removal_ind] + self.indices[removal_ind+1:]
        self.feature_metadata = self.feature_metadata[:removal_ind] + self.feature_metadata[removal_ind+1:]

        if not self.use_equal_weights:
            self.feature_weights = nn.Parameter(
                torch.cat([
                    self.feature_weights[:removal_ind],
                    self.feature_weights[removal_ind+1:]
                ])
            )

    def forward(self, all_features: torch.Tensor):
        # Extract features
        if self.copy_input_features:
            all_features = all_features.clone()

        if self.indices:
            selected_features = all_features[..., self.indices]
        else:
            selected_features = all_features[..., self.start_ind:self.end_ind]

        # Prepare weights
        feature_weights = self.feature_weights
        if self.weight_by_probs:
            feature_weights = 1 / self.n_features if self.use_equal_weights else feature_weights.softmax(dim=-1)

        # Compute product
        output = selected_features * self.feature_weights

        # Set indices to output
        if self.indices:
            all_features[..., self.indices] = output
        else:
            all_features[..., self.start_ind:self.end_ind] = output

        return output