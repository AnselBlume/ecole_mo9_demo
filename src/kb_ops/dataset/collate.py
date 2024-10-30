import torch
from torch import Tensor
from kb_ops.caching.cached_image_features import CachedImageFeatures
from dataclasses import fields
from typing import Any, Callable
import numpy as np

def list_collate(batch: list[dict]) -> dict[str, Any]:
    keys = batch[0].keys()

    return {k : [d[k] for d in batch] for k in keys}

REGION_TENSOR_NAMES = [
    'region_features',
    'clip_region_features',
    'region_weights',
    'trained_attr_region_scores'
]

class BatchCachedFeaturesCollate:
    def __init__(self, concept_names: list[str] = None):
        '''
            concept_names: List of concept names to stack across all dictionaries in CachedImageFeatures.
                If not provided, the intersection of concept names across all dictionaries will be used.
                Providing this argument speeds up the collation process and helps with memory for exceptionally large graphs.
        '''
        self.concept_names = concept_names

    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        batch = list_collate(batch)

        if 'features' not in batch:
            raise RuntimeError('Batch does not contain features.')

        features_list: list[CachedImageFeatures] = batch['features']
        assert all(isinstance(f, CachedImageFeatures) for f in features_list)

        # Batch features
        batched_features = CachedImageFeatures(is_batched=True)

        # For each tensor value in CachedImageFeatures, stack and set on batch features
        # NOTE Regions are concatenated along the -2'nd dimension instead of stacking as there are varying numbers of regions per image
        # This must be handled separately by the ConceptPredictor and is indicated by ImageFeatures.is_batched
        for field in fields(CachedImageFeatures):
            field_value = getattr(features_list[0], field.name)
            if isinstance(field_value, Tensor):
                if field.name in REGION_TENSOR_NAMES:
                    merge_fn = torch.cat
                else:
                    merge_fn = torch.stack

                self._merge_and_set_attribute(field.name, features_list, batched_features, merge_fn=merge_fn)

        # For each dictionary value in CachedImageFeatures, stack each concept's value and set on batch features
        batched_features.concept_to_zs_attr_img_scores = self._merge_tensor_dicts('concept_to_zs_attr_img_scores', features_list)
        batched_features.concept_to_zs_attr_region_scores = self._merge_tensor_dicts('concept_to_zs_attr_region_scores', features_list, merge_fn=torch.cat)
        batched_features.component_concept_scores = self._merge_tensor_dicts('component_concept_scores', features_list)

        batched_features.n_regions_per_image = [f.region_weights.shape[0] for f in features_list]

        batch['features'] = batched_features

        return batch

    def _merge_tensor_dicts(
        self,
        dict_attr_name: str,
        features: list[CachedImageFeatures],
        merge_fn: Callable[[list[Tensor]], Tensor] = torch.stack
    ) -> dict[str, Tensor]:

        if self.concept_names:
            concept_names = self.concept_names
        else:
            # Compute intersection of concept names across all dictionaries
            concept_names: set[str] = set(getattr(features[0], dict_attr_name).keys())
            for f in features[1:]:
                concept_names &= set(getattr(f, dict_attr_name).keys())

            concept_names: list[str] = sorted(concept_names) # Determinism

        merged_dict = {}
        for concept_name in concept_names:
            all_tensors = []
            exists_feature_without_concept = False
            exists_feature_with_concept = False

            for feature in features:
                feature_dict = getattr(feature, dict_attr_name)

                if concept_name not in feature_dict:
                    exists_feature_without_concept = True
                    break
                else:
                    exists_feature_with_concept = True
                    all_tensors.append(feature_dict[concept_name])

            # If some features have the concept and others don't, something went wrong
            if exists_feature_without_concept and exists_feature_with_concept:
                raise RuntimeError(f'Concept {concept_name} is missing in some features.')

            # If there does not exist a feature without the concept (i.e. all features have the concept), merge tensors and store
            if not exists_feature_without_concept:
                merged_dict[concept_name] = merge_fn(all_tensors)

        return merged_dict

    def _merge_and_set_attribute(
        self,
        attr_name: str,
        features_list: list[CachedImageFeatures],
        batched_features: CachedImageFeatures,
        merge_fn: Callable[[list[Tensor]], Tensor] = torch.stack
    ):

        attr = merge_fn([getattr(f, attr_name) for f in features_list])
        setattr(batched_features, attr_name, attr)

class BatchedCachedImageFeaturesIndexer:
    def __init__(
        self,
        batched_features: CachedImageFeatures,
        labels: list[str],
        batch_size: int,
        shuffle: bool = True,
        random_seed = None
    ):

        if not batched_features.is_batched:
            raise ValueError('BatchedCachedImageFeaturesIndexer requires a batched CachedImageFeatures object.')

        self.batched_features = batched_features
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed

        self.dataset_len = len(batched_features.image_features)
        self.rng = np.random.default_rng(random_seed)
        self.batch_indices = self._get_batch_indices()
        self.curr_batch_index = 0

        # Mapping from image index to the start index of its region features in the -2nd dimension of a regions tensor
        self.region_feature_start_inds = np.cumsum([0] + self.batched_features.n_regions_per_image)

    def _get_batch_indices(self) -> list[np.ndarray[int]]:
        '''
            Returns a list of numpy arrays of indices for each batch.
        '''
        indices = np.arange(self.dataset_len)
        if self.shuffle:
            self.rng.shuffle(indices)

        batch_indices = []
        for i in range(0, len(indices), self.batch_size):
            batch_indices.append(indices[i:i + self.batch_size])

        return batch_indices

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_batch_index >= len(self.batch_indices):
            raise StopIteration

        batch_indices = self.batch_indices[self.curr_batch_index]
        self.curr_batch_index += 1

        features = self._select_features_from_indices(batch_indices)
        labels = self.labels[batch_indices].tolist()

        return  {
            'features': features,
            'labels': labels
        }

    def __len__(self):
        return np.round(len(self.batched_features.image_features) / self.batch_size)

    def _select_features_from_indices(self, indices: np.ndarray[int]) -> CachedImageFeatures:
        '''
            Returns a CachedImageFeatures object with only the features corresponding to the given indices.
        '''
        selected_features = CachedImageFeatures(is_batched=True)

        for field in fields(CachedImageFeatures):
            field_value = getattr(self.batched_features, field.name)
            if isinstance(field_value, Tensor):
                if field.name in REGION_TENSOR_NAMES:
                    region_features = self._extract_region_features(field_value, indices)
                    setattr(selected_features, field.name, region_features)
                else:
                    setattr(selected_features, field.name, field_value[indices])

            elif isinstance(field_value, dict):
                setattr(selected_features, field.name, {k : v[indices] for k, v in field_value.items()})

        selected_features.n_regions_per_image = np.array(self.batched_features.n_regions_per_image)[indices].tolist()

        return selected_features

    def _extract_region_features(self, all_region_features: Tensor, indices: np.ndarray[int]) -> Tensor:
        '''
            Extracts region features from the given tensor for examples at the specified indices.

            In paticular, each image can have a variable number of regions specified by CachedImageFeatures.n_regions_per_image.
            Therefore, the region features tensor is concatenated along the 0'th dimension by the BatchCachedFeaturesCollate class
            and we must extract the regions for each image separately.

            Arguments:
                all_region_features: Tensor of shape (..., n_regions, d_regions) or (..., n_regions) of region features.
                indices: Indices of examples to extract region features for.
        '''
        extracted_region_features = []
        for image_ind in indices:
            start_ind = self.region_feature_start_inds[image_ind]
            end_ind = self.region_feature_start_inds[image_ind + 1]
            image_region_features = all_region_features[start_ind:end_ind, ...] # Assumes leading dimension is n_regions
            extracted_region_features.append(image_region_features)

        return torch.cat(extracted_region_features)