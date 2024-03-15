import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from feature_extraction import CLIPFeatureExtractor, CLIPTrainedAttributePredictor, CLIPAttributePredictor, DinoFeatureExtractor
from PIL.Image import Image
from model.features import ImageFeatures

class FeatureExtractor(nn.Module):
    # TODO batch the zs_attrs by making it a list of lists, flattening, then chunking
    def __init__(self, dino: nn.Module, clip: CLIPModel, processor: CLIPProcessor):
        super().__init__()

        self.dino = dino
        self.clip = clip
        self.processor = processor

        self.dino_feature_extractor = DinoFeatureExtractor(dino)
        self.clip_feature_extractor = CLIPFeatureExtractor(clip, processor)
        self.trained_attr_predictor = CLIPTrainedAttributePredictor(self.clip_feature_extractor)
        self.zs_attr_predictor = CLIPAttributePredictor(clip, processor)

    def forward(
        self,
        image: Image,
        regions: list[Image],
        zs_attrs: list[str],
        cached_visual_features: torch.Tensor = None,
        cached_trained_attr_scores: torch.Tensor = None,
        cached_clip_visual_features: torch.Tensor = None
    ):
        '''
            If cached_visual_features is provided, does not recompute image and region features.
            Should have shape (1 + n_regions, d_img) where the first element is the image feature.

            If cached_trained_attr_scores is provided, does not recompute trained attribute scores.
            Should have shape (1 + n_regions, n_learned_attrs) where the first element is the image feature.
        '''
        # DINO image features
        if cached_visual_features is None:
            visual_features = self.dino_feature_extractor([image] + regions)[0] # CLS features, (1 + n_regions, d_img)
        else:
            visual_features = cached_visual_features

        # CLIP image features
        device = visual_features.device

        if cached_clip_visual_features is None:
            clip_visual_features = self.clip_feature_extractor(images=[image] + regions)
        else:
            clip_visual_features = cached_clip_visual_features

        # Zero-shot attributes from CLIP features
        if len(zs_attrs):
            zs_features = self.clip_feature_extractor(texts=zs_attrs)
            zs_scores = self.zs_attr_predictor.feature_score(clip_visual_features, zs_features) # (1 + n_regions, n_zs_attrs)
        else:
            zs_scores = torch.tensor([[]], device=device) # This will be a nop in the indexing below

        # Trained attribute scores from CLIP, soon to be DINO features
        if cached_trained_attr_scores is None:
            if len(self.trained_attr_predictor.attr_names):
                trained_attr_scores = self.trained_attr_predictor.predict_from_features(clip_visual_features) # (1 + n_regions, n_learned_attrs)
            else:
                trained_attr_scores = torch.tensor([[]], device=device) # (1, 0); nop in the indexing below
        else:
            trained_attr_scores = cached_trained_attr_scores

        region_weights = torch.ones(len(regions), device=device) / len(regions) # Uniform weights

        return ImageFeatures(
            image_features=visual_features[:1], # (1, d_img)
            clip_image_features=clip_visual_features[:1], # (1, d_img)
            region_features=visual_features[1:], # (n_regions, d_img)
            clip_region_features=clip_visual_features[1:], # (n_regions, d_img)
            region_weights=region_weights,
            trained_attr_img_scores=trained_attr_scores[:1],
            trained_attr_region_scores=trained_attr_scores[1:],
            zs_attr_img_scores=zs_scores[:1],
            zs_attr_region_scores=zs_scores[1:],
        )

