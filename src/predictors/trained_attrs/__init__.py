# %%
import json
import torch
import torch.nn as nn
from PIL.Image import Image
import os
from typing import Optional
from predictors.clip_features import CLIPFeatureExtractor

DEFAULT_CKPT_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../attribute_training/classifiers_bias/classifiers.pth'
)

with open(os.path.join(os.path.dirname(__file__), 'attribute_index.json')) as f:
    attr_to_index = json.load(f)

INDEX_TO_ATTR = {v: k for k, v in attr_to_index.items()}

class TrainedCLIPAttributePredictor:
    def __init__(self, clip_feature_extractor: CLIPFeatureExtractor, ckpt_path=DEFAULT_CKPT_PATH, device='cuda'):
        self.clip_feature_extractor = clip_feature_extractor.eval()

        # Load and stack classifiers for efficiency
        classifiers: nn.ModuleList[nn.Linear] = torch.load(ckpt_path, map_location=device) # Each linear weight has shape (1, enc_dim)

        with torch.no_grad():
            stacked_weight = torch.stack([classifier.weight.squeeze() for classifier in classifiers]) # (num_cls, enc_dim)
            stacked_bias = torch.cat([classifier.bias for classifier in classifiers]) # (num_cls,)

        self.predictor = nn.Linear(stacked_weight.shape[1], stacked_weight.shape[0], bias=True)
        self.predictor.weight = nn.Parameter(stacked_weight)
        self.predictor.bias = nn.Parameter(stacked_bias)

    @torch.inference_mode()
    def predict(self, image: Image, apply_sigmoid: bool = True, threshold: Optional[float] = 0.5):
        '''
            Returns a torch.Tensor with shape (1, num_cls) of scores in if threshold is None.
            Otherwise, returns a torch.BoolTensor with shape (1, num_cls) indicating values above threshold.
        '''
        # Compute scores
        img_feats = self.clip_feature_extractor(image=image).float()

        if img_feats.device != self.predictor.weight.device:
            img_feats = img_feats.to(self.predictor.weight.device)

        preds = self.predictor(img_feats)

        # Potentially apply sigmoid and threshold
        if apply_sigmoid:
            preds = preds.sigmoid()

        if threshold is not None:
            assert apply_sigmoid, 'Thresholding only supported when apply_sigmoid is True'
            assert 0 <= threshold <= 1, 'Threshold must be between 0 and 1'
            preds = preds > threshold

        return preds
# %%
