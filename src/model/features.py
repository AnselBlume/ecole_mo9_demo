from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass, field
from typing import Optional
from model.dataclass_base import DeviceShiftable
import logging

logger = logging.getLogger(__name__)

import torch
from dataclasses import dataclass

@dataclass
class ImageFeatures(DeviceShiftable):
    #####################################
    # Features for internal calculation #
    #####################################
    image_features: Tensor = None # (..., 1, d_img)
    clip_image_features: Tensor = None # (..., 1, d_img)

    # NOTE if this ImageFeatures is batched, the n_regions dimension will enumerate regions for all
    # images in the batch, and this must be handled separately by splitting via n_regions_per_image
    region_features: Tensor = None # (..., n_regions, d_regions)
    clip_region_features: Tensor = None # (..., n_regions, d_regions)
    region_weights: Tensor = None # (..., n_regions); how much to weight each region in all calculations

    trained_attr_img_scores: Tensor = None # (..., 1, n_trained_attrs)
    trained_attr_region_scores: Tensor = None # (..., n_regions, n_trained_attrs)

    # Whether features are batched
    # If is_batched, all region tensors will be concatenated along the -2'nd dimension instead of stacked
    # due to the variable number of regions per image. n_regions_per_images must be set in this case
    # In this case, dimension -2 of region tensors will be the sum of the number of regions of each image in the batch
    is_batched: bool = False # Whether the features are batched.
    n_regions_per_image: list[int] = None # Number of regions per image in the batch

    #############################################
    # Features computed via batched calculation #
    #############################################
    # Tensor of shape (1 + 1 + 2*n_learned_attrs + 2*n_zs_attrs,) where the first and second elmts are
    # the image and region scores, respectively
    all_scores: Tensor = None

    def validate_dimensions(self):
        if self.is_batched:
            bsize = len(self.image_features)

            self._validate_leading_dimension('image_features', bsize)
            self._validate_leading_dimension('clip_image_features', bsize)
            self._validate_leading_dimension('trained_attr_img_scores', bsize)

            if self.region_weights is not None:
                assert self.n_regions_per_image is not None, 'n_regions_per_image must be set if is_batched is True'
                n_total_regions = sum(self.n_regions_per_image)

                self._validate_leading_dimension('region_features', n_total_regions)
                self._validate_leading_dimension('clip_region_features', n_total_regions)
                self._validate_leading_dimension('region_weights', n_total_regions)
                self._validate_leading_dimension('trained_attr_region_scores', n_total_regions)

        else:
            self._validate_leading_dimension('image_features', 1)
            self._validate_leading_dimension('clip_image_features', 1)
            self._validate_leading_dimension('trained_attr_img_scores', 1)

            if self.region_weights is not None:
                n_regions = self.region_weights.shape[0]

                self._validate_leading_dimension('region_features', n_regions)
                self._validate_leading_dimension('clip_region_features', n_regions)
                self._validate_leading_dimension('trained_attr_region_scores', n_regions)


    def _validate_leading_dimension(self, tensor_attr_name: str, expected_shape: int):
        tensor: Optional[Tensor] = getattr(self, tensor_attr_name)
        assert tensor is None or tensor.shape[0] == expected_shape, (
            f'{tensor_attr_name} must have leading dimension of {expected_shape}'
        )

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
        features.is_batched = image_features.is_batched
        features.n_regions_per_image = image_features.n_regions_per_image

        return features

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

    def forward(self, all_features: Tensor):
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