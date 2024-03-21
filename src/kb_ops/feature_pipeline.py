import torch
from image_processing import LocalizerAndSegmenter, LocalizeAndSegmentOutput
from model.concept import ConceptKB
from model.features import ImageFeatures
from feature_extraction import FeatureExtractor
from typing import Union, Optional
from PIL.Image import Image

class ConceptKBFeaturePipeline:
    def __init__(
        self,
        concept_kb: ConceptKB,
        loc_and_seg: LocalizerAndSegmenter,
        feature_extractor: FeatureExtractor
    ):
        self.concept_kb = concept_kb
        self.loc_and_seg = loc_and_seg
        self.feature_extractor = feature_extractor

    def get_segmentations(
        self,
        image: Image,
        concept_name: str = '',
        concept_parts: list[str] = [],
        remove_background: bool = True,
        return_crops: bool = True,
        use_bbox_for_crops: bool = False
    ) -> LocalizeAndSegmentOutput:

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

    def get_features(
        self,
        image: Image,
        segmentations: LocalizeAndSegmentOutput,
        zs_attrs: list[str],
        cached_features: ImageFeatures = None
    ) -> ImageFeatures:
        # Get region crops
        region_crops = segmentations.part_crops
        region_masks = segmentations.part_masks

        if region_crops == []:
            assert len(region_masks) == 0
            region_crops = [image]
            region_masks = torch.ones(1, *image.size[::-1], dtype=torch.bool)

        with torch.no_grad():
            features: ImageFeatures = self.feature_extractor(
                image,
                region_crops,
                zs_attrs,
                region_masks,
                cached_features=cached_features
            )

        return features