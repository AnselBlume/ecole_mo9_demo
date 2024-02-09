# %%
import json
import torch
import torch.nn as nn
import clip
from PIL.Image import Image
import os

DEFAULT_CKPT_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../attribute_training/classifiers_bias/classifiers.pth'
)

with open(os.path.join(os.path.dirname(__file__), 'attribute_index.json')) as f:
    attr_to_index = json.load(f)

INDEX_TO_ATTR = {v: k for k, v in attr_to_index.items()}

class CLIPImageFeatureExtractor(nn.Module):
    def __init__(self, model_name, device='cuda'):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)

    def forward(self, image: Image):
        '''
            image: PIL.Image.Image
            Returns: torch.Tensor of shape (1, enc_dim)
        '''
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        return self.model.encode_image(image_input)

class TrainedCLIPAttributePredictor:
    def __init__(self, clip_model_name='ViT-L/14', ckpt_path=DEFAULT_CKPT_PATH, device='cuda'):
        self.clip_feature_extractor = CLIPImageFeatureExtractor(model_name=clip_model_name, device=device)

        # Load and stack classifiers for efficiency
        classifiers: nn.ModuleList[nn.Linear] = torch.load(ckpt_path, map_location=device) # Each linear weight has shape (1, enc_dim)

        with torch.no_grad():
            stacked_weight = torch.stack([classifier.weight.squeeze() for classifier in classifiers]) # (num_cls, enc_dim)
            stacked_bias = torch.cat([classifier.bias for classifier in classifiers]) # (num_cls,)

        self.predictor = nn.Linear(stacked_weight.shape[1], stacked_weight.shape[0], bias=True)
        self.predictor.weight = nn.Parameter(stacked_weight)
        self.predictor.bias = nn.Parameter(stacked_bias)

    @torch.inference_mode()
    def predict(self, image: Image, threshold=0.5):
        '''
            Returns a torch.BoolTensor of shape (1, num_cls)
        '''
        img_feats = self.clip_feature_extractor(image).float()
        preds = self.predictor(img_feats).sigmoid() > threshold

        return preds
# %%
