import torch
from image_processing import LocalizerAndSegmenter, LocalizeAndSegmentOutput
from model.concept import Concept, ConceptPredictorFeatures
from model.features import ImageFeatures
from kb_ops.caching import CachedImageFeatures
from feature_extraction import FeatureExtractor
from typing import Union
from PIL.Image import Image
from dataclasses import dataclass

@dataclass
class ConceptKBFeaturePipelienConfig:
    compute_component_concept_scores: bool = False

    remove_background: bool = True
    return_crops: bool = True
    use_bbox_for_crops: bool = False

class ConceptKBFeaturePipeline:
    def __init__(
        self,
        loc_and_seg: LocalizerAndSegmenter,
        feature_extractor: FeatureExtractor,
        config: ConceptKBFeaturePipelienConfig = ConceptKBFeaturePipelienConfig()
    ):
        self.loc_and_seg = loc_and_seg
        self.feature_extractor = feature_extractor
        self.config = config

    def get_segmentations(
        self,
        image: Image,
        concept_name: str = '',
        concept_parts: list[str] = [],
        remove_background: bool = None,
        return_crops: bool = None,
        use_bbox_for_crops: bool = None
    ) -> LocalizeAndSegmentOutput:

        if remove_background is None:
            remove_background = self.config.remove_background
        if return_crops is None:
            return_crops = self.config.return_crops
        if use_bbox_for_crops is None:
            use_bbox_for_crops = self.config.use_bbox_for_crops

        return self.loc_and_seg.localize_and_segment(
            image=image,
            concept_name=concept_name,
            concept_parts=concept_parts,
            remove_background=remove_background,
            return_crops=return_crops,
            use_bbox_for_crops=use_bbox_for_crops
        )

    def get_image_and_segmentations(
        self,
        image_data: Union[Image, LocalizeAndSegmentOutput],
        **seg_kwargs
    ) -> tuple[Image, LocalizeAndSegmentOutput]:

        if isinstance(image_data, Image):
            segmentations = self.get_segmentations(image_data, **seg_kwargs)
            image = image_data

        else:
            # Check __name__ instead of isinstance to avoid pickle versioning issues
            assert image_data.__class__.__name__ == 'LocalizeAndSegmentOutput', f'Expected LocalizeAndSegmentOutput, got {type(image_data)}'
            segmentations = image_data
            image = segmentations.input_image

        return image, segmentations

    def get_image_features(
        self,
        image: Image,
        segmentations: LocalizeAndSegmentOutput,
        cached_features: ImageFeatures = None
    ) -> ImageFeatures:

        # Get region crops
        region_crops = segmentations.part_crops
        region_masks = segmentations.part_masks

        if region_crops == [] and not cached_features:
            assert len(region_masks) == 0
            region_crops = [image]
            region_masks = torch.ones(1, *image.size[::-1], dtype=torch.bool)

        with torch.no_grad():
            features: ImageFeatures = self.feature_extractor(
                image,
                region_crops,
                segmentations.object_mask,
                region_masks,
                cached_features=cached_features
            )

        return features

    def get_concept_predictor_features(
        self,
        image: Image,
        segmentations: LocalizeAndSegmentOutput,
        concept: Concept,
        cached_features: CachedImageFeatures = None,
    ) -> ConceptPredictorFeatures:
        '''
            Args:
                update_cached_features (bool): If true, updates the CachedImageFeatures with computed zero-shot attributes and component concept scores.
                    Component concept scores are computed and updated only if self.config.compute_component_concept_scores is True.
        '''

        if cached_features is None:
            # Base image features
            image_features = self.get_image_features(image, segmentations, cached_features)
            cached_features = CachedImageFeatures.from_image_features(image_features)

        # Zero-shot attribute scores
        zs_attr_img_scores, zs_attr_region_scores = self._get_zero_shot_attr_scores(concept, cached_features)

        # Component concept scores
        if self.config.compute_component_concept_scores:
            component_concept_scores = self._get_component_concept_scores(concept, cached_features)

        # Construct output
        concept_predictor_features = ConceptPredictorFeatures.from_image_features(cached_features)
        concept_predictor_features.zs_attr_img_scores = zs_attr_img_scores
        concept_predictor_features.zs_attr_region_scores = zs_attr_region_scores
        concept_predictor_features.component_concept_scores = component_concept_scores if self.config.compute_component_concept_scores else None

        return concept_predictor_features

    def _get_zero_shot_attr_scores(self, concept: Concept, cached_features: CachedImageFeatures):
        '''
            Returns a tuple of tensors of shapes:
                (1, n_zs_attrs) for the zero-shot attribute scores for the image.
                (n_regions, n_zs_attrs) for the zero-shot attribute scores for each region.
        '''
        if concept.name in cached_features.concept_to_zs_attr_img_scores and concept.name in cached_features.concept_to_zs_attr_region_scores:
            zs_attr_img_scores = cached_features.concept_to_zs_attr_img_scores[concept.name] # (1, n_zs_attrs)
            zs_attr_region_scores = cached_features.concept_to_zs_attr_region_scores[concept.name] # (n_regions, n_zs_attrs)

        else: # Need to compute
            clip_visual_features = self.feature_extractor._get_clip_visual_features(None, None, cached_features) # Extract instead of recomputing
            zs_attrs = [attr.query for attr in concept.zs_attributes]
            zs_attr_scores = self.feature_extractor.get_zero_shot_attr_scores(clip_visual_features, zs_attrs)

            zs_attr_img_scores = zs_attr_scores[:1]
            zs_attr_region_scores = zs_attr_scores[1:]

        return zs_attr_img_scores, zs_attr_region_scores

    def _get_component_concept_scores(
        self,
        concept: Concept,
        cached_features: CachedImageFeatures = None,
        image: Image = None,
        recompute_existing: bool = False
    ) -> torch.Tensor:

        '''
            Gets the component concept scores for a given concept using DesCo. Note that if one wishes to compute the component concept scores
            using the ConceptPredictor, that must be managed externally.

            Args:
                concept (Concept): The concept for which to compute the component concept scores.
                cached_features (ImageFeatures): The cached features to use for computing the scores.
                image (Image): The image to compute the scores for. Used only if image features are not present in cached_features.
                recompute_existing (bool): Whether to recompute the scores for concepts that are already present in the cache. Takes longer
                    as needs to compute scores for all component concepts, instead of just the ones without scores.

            Returns:
                Tensor of shape (1, n_component_concepts) containing the scores for each component concept.
        '''

        if not len(concept.component_concepts):
            return torch.tensor([])

        # Determine which concepts we need to compute scores for
        cached_features = CachedImageFeatures() if cached_features is None else cached_features
        concept_to_scores = dict(cached_features.component_concept_scores) # Make a copy so don't cause side-effects

        if recompute_existing:
            concept_scores_to_compute = concept.component_concepts
        else:
            concept_scores_to_compute = [concept_name for concept_name in concept.component_concepts if concept_name not in concept_to_scores]

        new_concept_scores = self.feature_extractor.get_component_scores(image, concept.component_concepts.keys())

        concept_to_scores.update({concept_name : score for concept_name, score in zip(concept_scores_to_compute, new_concept_scores)})

        # Extract scores
        scores = [concept_to_scores[component_concept_name] for component_concept_name in concept.component_concepts]
        scores = torch.stack(scores) # (1, n_component_concepts)

        return scores