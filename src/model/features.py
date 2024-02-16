import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional
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

    region_features: torch.Tensor = None # (n_regions, d_regions)
    region_weights: torch.Tensor = None # (n_regions,); how much to weight each region in all calculations

    trained_attr_img_scores: torch.Tensor = None # (1, n_trained_attrs)
    trained_attr_region_scores: torch.Tensor = None # (n_regions, n_trained_attrs,)

    zs_attr_img_scores: torch.Tensor = None # (1, n_zs_attrs)
    zs_attr_region_scores: torch.Tensor = None # (n_regions, n_zs_attrs)

    #############################################
    # Features computed via batched calculation #
    #############################################
    # Tensor of shape (1 + 1 + n_learned_attrs + n_zs_attrs,) where the first and second elmts are
    # the image and region scores, respectively
    all_scores: torch.Tensor = None

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