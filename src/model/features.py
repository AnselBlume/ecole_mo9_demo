import torch
import torch.nn as nn
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

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
        weight_by_probs: bool = False,
        copy_input_features: bool = True,
        feature_metadata: list[FeatureMetadata] = None
    ):
        '''
            indices (Optional[list[int]]): Indicates the specific indices to use from the input feature vector.
            range (Optional[tuple[int,int]]) Indicates the range of indices to use from the input feature vector (end index exclusive).
            weight_by_probs (bool): Whether to normalize learned weights into probabilities before multiplying.
            copy_input_features (bool): Whether to copy the input tensor before modifying it (to avoid modifying the original tensor in-place).
            feature_metadata (Optional[list[FeatureMetadata]]): Metadata for each feature in the group.
        '''
        super().__init__()

        assert (range is None) ^ (indices is None), 'Exactly one of range or indices must be provided.'

        self.start_ind, self.end_ind = range if range else (None, None)
        self.indices = indices

        self.weight_by_probs = weight_by_probs
        self.copy_input_features = copy_input_features

        self.n_features = len(indices) if indices else (self.end_ind - self.start_ind)
        self.feature_weights = nn.Parameter(torch.ones(self.n_features) / self.n_features)

        assert feature_metadata is None or len(feature_metadata) == self.n_features
        self.feature_metadata = feature_metadata

    def _convert_to_index_based(self):
        self.indices = list(range(self.start_ind, self.end_ind))
        self.start_ind, self.end_ind = None, None

    def add_weight(self, index: int, value: float = None):
        if value is None:
            value = torch.mean(self.feature_weights).item()

        if not self.indices: # Range-based
            logger.info('Feature group is range-based and insertion would shift weight indices; converting to index-based.')
            self._convert_to_index_based()

        self.indices.append(index)
        self.feature_weights = nn.Parameter([
            torch.cat([self.feature_weights, torch.tensor([value])])
        ])

    def remove_weight(self, index: int):
        if not self.indices:
            logger.info('Feature group is range-based and removal would shift weight indices; converting to index-based.')
            self._convert_to_index_based()

        self.indices.remove(index)
        self.feature_weights = nn.Parameter(
            torch.cat([self.feature_weights[:index], self.feature_weights[index+1:]])
        )

    def forward(self, all_features: torch.Tensor):
        assert input.shape[-1] == self.n_features

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
            feature_weights = feature_weights.softmax(dim=-1)

        # Compute product
        output = selected_features * self.feature_weights

        # Set indices to output
        if self.indices:
            all_features[..., self.indices] = output
        else:
            all_features[..., self.start_ind:self.end_ind] = output

        return output