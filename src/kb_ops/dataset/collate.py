import torch
from torch import Tensor
from kb_ops.caching.cached_image_features import CachedImageFeatures
from dataclasses import fields
from typing import Any, Callable

def list_collate(batch: list[dict]) -> dict[str, Any]:
    keys = batch[0].keys()

    return {k : [d[k] for d in batch] for k in keys}

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
                if field.name in ['region_features', 'clip_region_features', 'region_weights', 'trained_attr_region_scores']:
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

        return {
            concept_name : merge_fn([getattr(f, dict_attr_name)[concept_name] for f in features])
            for concept_name in concept_names
        }

    def _merge_and_set_attribute(
            self,
            attr_name: str,
            features_list: list[CachedImageFeatures],
            batched_features: CachedImageFeatures,
            merge_fn: Callable[[list[Tensor]], Tensor] = torch.stack
        ):

        attr = merge_fn([getattr(f, attr_name) for f in features_list])
        setattr(batched_features, attr_name, attr)