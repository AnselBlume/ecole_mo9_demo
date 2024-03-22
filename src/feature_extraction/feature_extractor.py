import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from feature_extraction import (
    CLIPFeatureExtractor,
    CLIPTrainedAttributePredictor,
    CLIPAttributePredictor,
    DINOFeatureExtractor,
    DINOTrainedAttributePredictor
)
from PIL.Image import Image
from model.features import ImageFeatures
from feature_extraction.dino_features import get_rescaled_features, region_pool, interpolate_masks
import logging
logger = logging.getLogger(__file__)

class FeatureExtractor(nn.Module):
    # TODO batch the zs_attrs by making it a list of lists, flattening, then chunking
    def __init__(self, dino: nn.Module, clip: CLIPModel, processor: CLIPProcessor):
        super().__init__()

        self.dino = dino
        self.clip = clip
        self.processor = processor

        # Can't resize DINO images as region masks won't correspond to image size, unless we resize the masks as well
        self.dino_feature_extractor = DINOFeatureExtractor(dino, crop_images=False)
        self.clip_feature_extractor = CLIPFeatureExtractor(clip, processor)
        self.trained_attr_predictor = DINOTrainedAttributePredictor(self.dino_feature_extractor, device=self.dino_feature_extractor.device)
        self.zs_attr_predictor = CLIPAttributePredictor(clip, processor)

    def forward(
        self,
        image: Image,
        regions: list[Image],
        zs_attrs: list[str],
        object_mask: torch.BoolTensor,
        region_masks: torch.BoolTensor,
        cached_features: ImageFeatures = None
    ):
        '''
            If cached_visual_features is provided, does not recompute image and region features.
            Should have shape (1 + n_regions, d_img) where the first element is the image feature.

            If cached_trained_attr_scores is provided, does not recompute trained attribute scores.
            Should have shape (1 + n_regions, n_learned_attrs) where the first element is the image feature.
        '''
        cached_features = ImageFeatures() if cached_features is None else cached_features # Prevent None checking of object

        # DINO image features
        dino_device = self.dino_feature_extractor.device

        if None in [cached_features.image_features, cached_features.region_features]:
            # Generate DINO features

            # try: # Variant using CLS features for image and regions
            #     cls_features = self.dino_feature_extractor([image] + regions)[0]

            # except RuntimeError:
            #     logger.info(f'Out of memory on DINO forward for image with size {image.size[::-1]} with {len(regions)} regions; performing forward individually')
            #     cls_features = []
            #     for img in [image] + regions:
            #         cls_features.append(self.dino_feature_extractor([img])[0])

            #     cls_features = torch.cat(cls_features, dim=0)

            # image_features, region_features = cls_features[:1], cls_features[1:]

            rescale_kwargs = {
                'feature_extractor': self.dino_feature_extractor,
                'images': [image],
                'patch_size': self.dino_feature_extractor.model.patch_size,
            }

            is_on_cpu = False # Whether getting rescaled features fails and we fall back to CPU

            try:
                image_features, patch_features = get_rescaled_features(**rescale_kwargs)

            except RuntimeError:
                logger.info('Ran out of memory rescaling patch features; falling back to CPU')
                torch.cuda.empty_cache()
                image_features, patch_features = get_rescaled_features(**rescale_kwargs, interpolate_on_cpu=True, return_on_cpu=True)
                is_on_cpu = True

            # DINOFeatureExtractor returns a list for patch features if not resizing image, which we don't to region pool
            if isinstance(patch_features, list):
                patch_features = patch_features[0].unsqueeze(0) # (1, h, w, d)

            # TODO include image foreground mask and use image_features as pooled features

            # Potentially resize or crop region masks to match patch features
            region_masks = interpolate_masks(
                region_masks,
                do_resize=self.dino_feature_extractor.resize_images,
                do_crop=self.dino_feature_extractor.crop_images
            ) # (n, h, w)

            # Perform region pooling
            # We allow empty masks here, which can arise when a region is cropped out or resized to zero (based on the DinoFeatureExtractor transform).
            # Empty masks return a zero region feature vector
            try:
                region_features = region_pool(region_masks.to(patch_features.device), patch_features, allow_empty_masks=True) # (n, d)

            except RuntimeError:
                logger.info('Ran out of memory region pooling; falling back to CPU')
                torch.cuda.empty_cache()
                region_features = region_pool(region_masks.cpu(), patch_features.cpu(), allow_empty_mask=True) # (n, d)
                is_on_cpu = True

            if is_on_cpu:
                image_features = image_features.to(dino_device)
                region_features = region_features.to(dino_device)

        else:
            image_features = cached_features.image_features
            region_features = cached_features.region_features

        visual_features = torch.cat([image_features, region_features], dim=0) # (1 + n_regions, d_img)

        # CLIP image features
        clip_device = self.clip_feature_extractor.device

        if None in [cached_features.clip_image_features, cached_features.clip_region_features]:
            clip_visual_features = self.clip_feature_extractor(images=[image] + regions)
        else:
            clip_visual_features = torch.cat([cached_features.clip_image_features, cached_features.clip_region_features], dim=0)

        # Zero-shot attributes from CLIP features
        if len(zs_attrs):
            zs_features = self.clip_feature_extractor(texts=zs_attrs)
            zs_scores = self.zs_attr_predictor.feature_score(clip_visual_features, zs_features) # (1 + n_regions, n_zs_attrs)
        else:
            zs_scores = torch.tensor([[]], device=clip_device) # This will be a nop in the indexing below

        # Trained attribute scores from DINO features
        if None in [cached_features.trained_attr_img_scores, cached_features.trained_attr_region_scores]:
            if len(self.trained_attr_predictor.attr_names):
                # NOTE We take the sigmoid for DINO as the feature, but not for CLIP since CLIP wasn't trained with BCELoss
                trained_attr_scores = self.trained_attr_predictor.predict_from_features(visual_features).sigmoid() # (1 + n_regions, n_learned_attrs)
            else:
                trained_attr_scores = torch.tensor([[]], device=dino_device) # (1, 0); nop in the indexing below
        else:
            trained_attr_scores = torch.cat([cached_features.trained_attr_img_scores, cached_features.trained_attr_region_scores], dim=0)

        region_weights = torch.ones(len(regions), device=clip_device) / len(regions) # Uniform weights

        return ImageFeatures(
            image_features=image_features, # (1, d_img)
            clip_image_features=clip_visual_features[:1], # (1, d_img)
            region_features=region_features, # (n_regions, d_img)
            clip_region_features=clip_visual_features[1:], # (n_regions, d_img)
            region_weights=region_weights,
            trained_attr_img_scores=trained_attr_scores[:1],
            trained_attr_region_scores=trained_attr_scores[1:],
            zs_attr_img_scores=zs_scores[:1],
            zs_attr_region_scores=zs_scores[1:],
        )

