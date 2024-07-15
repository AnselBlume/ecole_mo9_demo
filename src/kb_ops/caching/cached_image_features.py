import torch
from model.concept import ConceptPredictorFeatures, Concept
from model.features import ImageFeatures
from dataclasses import dataclass, field

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
            component_concept_scores=self.component_concept_scores.get(concept_name, None),
            is_batched=self.is_batched,
            n_regions_per_image=self.n_regions_per_image
        )

    def __getitem__(self, concept_name: str):
        return self.get_concept_predictor_features(concept_name)

    def update_concept_predictor_features(self, concept: Concept, features: ConceptPredictorFeatures, store_component_concept_scores: bool = True):
        self.concept_to_zs_attr_img_scores[concept.name] = features.zs_attr_img_scores.cpu()
        self.concept_to_zs_attr_region_scores[concept.name] = features.zs_attr_region_scores.cpu()

        if store_component_concept_scores:
            if not features.component_concept_scores:
                return

            component_concept_scores = features.component_concept_scores.cpu()
            assert len(concept.component_concepts) == len(component_concept_scores)

            # Overwriting existing scores is okay because the score is fixed for a given image-concept pair
            for component_concept_name, score in zip(concept.component_concepts, component_concept_scores):
                self.component_concept_scores[component_concept_name] = score

        return self

