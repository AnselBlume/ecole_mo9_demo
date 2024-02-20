import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from feature_extraction import CLIPFeatureExtractor, TrainedCLIPAttributePredictor, CLIPAttributePredictor
from PIL.Image import Image
from model.features import ImageFeatures

class FeatureExtractor(nn.Module):
    # TODO batch the zs_attrs by making it a list of lists, flattening, then chunking
    def __init__(self, clip: CLIPModel, processor: CLIPProcessor):
        super().__init__()

        self.clip = clip
        self.processor = processor

        self.clip_feature_extractor = CLIPFeatureExtractor(clip, processor)
        self.trained_clip_attr_predictor = TrainedCLIPAttributePredictor(self.clip_feature_extractor)
        self.zs_attr_predictor = CLIPAttributePredictor(clip, processor)

    def forward(
        self,
        image: Image,
        regions: list[Image],
        zs_attrs: list[str],
        cached_visual_features: torch.Tensor = None,
        cached_trained_attr_scores: torch.Tensor = None
    ):
        '''
            If cached_visual_features is provided, does not recompute image and region features.
            Should have shape (1 + n_regions, d_img) where the first element is the image feature.

            If cached_trained_attr_scores is provided, does not recompute trained attribute scores.
            Should have shape (1 + n_regions, n_learned_attrs) where the first element is the image feature.
        '''
        if cached_visual_features is None:
            visual_features = self.clip_feature_extractor(images=[image] + regions)
        else:
            visual_features = cached_visual_features

        img_features = visual_features[:1] # (1, d_img)
        region_features = visual_features[1:] # (n_regions, d_img)

        if len(zs_attrs):
            zs_features = self.clip_feature_extractor(texts=zs_attrs)
            zs_scores = self.zs_attr_predictor.feature_score(visual_features, zs_features) # (1 + n_regions, n_zs_attrs)
        else:
            zs_scores = torch.tensor([[]]) # This will be a nop in the indexing below

        if cached_trained_attr_scores is None:
            trained_attr_scores = self.trained_clip_attr_predictor.predict_from_features(visual_features) # (1 + n_regions, n_learned_attrs)
        else:
            trained_attr_scores = cached_trained_attr_scores

        region_weights = torch.ones(len(regions), device=trained_attr_scores.device) / len(regions) # Uniform weights

        return ImageFeatures(
            image_features=img_features,
            region_features=region_features,
            region_weights=region_weights,
            trained_attr_img_scores=trained_attr_scores[:1],
            trained_attr_region_scores=trained_attr_scores[1:],
            zs_attr_img_scores=zs_scores[:1],
            zs_attr_region_scores=zs_scores[1:],
        )

