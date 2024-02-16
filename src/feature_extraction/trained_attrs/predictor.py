import torch
import torch.nn as nn
from PIL.Image import Image
from typing import Optional
from feature_extraction.clip_features import CLIPFeatureExtractor
import torch.linalg as LA

from . import DEFAULT_CKPT_PATH, COLOR_SHAPE_MATERIAL_SUBSET, INDEX_TO_ATTR

class TrainedCLIPAttributePredictor:
    def __init__(
        self,
        clip_feature_extractor: CLIPFeatureExtractor,
        use_subset: bool = True,
        ckpt_path=DEFAULT_CKPT_PATH,
        device='cuda',
    ):
        self.clip_feature_extractor = clip_feature_extractor.eval()

        # Load all classifiers and potentially select a subset of attributes
        classifiers: nn.ModuleList[nn.Linear] = torch.load(ckpt_path, map_location=device) # Each linear weight has shape (1, enc_dim)

        if use_subset:
            classifier_subset = []
            attr_names = []

            for k in ['color', 'shape', 'material']:
                att_to_ind = COLOR_SHAPE_MATERIAL_SUBSET[k]
                for attr, ind in att_to_ind.items():
                    classifier_subset.append(classifiers[ind])
                    attr_names.append(attr)

            classifiers = classifier_subset

        else: # Use all attributes
            attr_names = list(INDEX_TO_ATTR.keys())

        self.attr_names = attr_names

        # Stack classifiers for efficiency
        with torch.no_grad():
            stacked_weight = torch.stack([classifier.weight.squeeze() for classifier in classifiers]) # (num_cls, enc_dim)
            stacked_weight = stacked_weight / LA.norm(stacked_weight, dim=-1, keepdim=True) # Pre-normalize to avoid in forward
            stacked_bias = torch.cat([classifier.bias for classifier in classifiers]) # (num_cls,)

        self.predictor = nn.Linear(stacked_weight.shape[1], stacked_weight.shape[0], bias=True)
        self.predictor.weight = nn.Parameter(stacked_weight.float()) # Loaded in double, so cast to float
        self.predictor.bias = nn.Parameter(stacked_bias.float())

    @torch.no_grad()
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

    @torch.no_grad()
    def predict_from_features(self, img_feats: torch.Tensor):
        '''
            img_feats: torch.Tensor of shape (n_imgs, d)

            Returns a torch.Tensor with shape (n_imgs, num_cls) of scores in [-1, 1]
        '''
        return self.predictor(img_feats) / LA.norm(img_feats, dim=-1, keepdim=True)