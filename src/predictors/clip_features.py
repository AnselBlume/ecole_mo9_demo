import clip
from clip.model import CLIP
from PIL import Image
import torch.nn as nn
from torchvision.transforms import Compose

class CLIPFeatureExtractor(nn.Module):
    def __init__(self, model: CLIP, preprocess: Compose):
        super().__init__()

        self.model = model.eval()
        self.preprocess = preprocess

    def forward(self, *, image: Image = None, texts: list[str] = None):
        '''
            image: PIL.Image.Image
            texts: list[str]
            Returns:
                If only image is provided, torch.Tensor of shape (1, enc_dim).
                If only texts is provided, torch.Tensor of shape (n_texts, enc_dim).
                If both are provided, returns a tuple of (image_feats, text_feats).
        '''
        assert image is not None or texts is not None, 'At least one of image or texts must be provided'

        ret_vals = []

        if image is not None:
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            image_feats = self.model.encode_image(image_input)
            ret_vals.append(image_feats)

        if texts is not None:
            text_input = clip.tokenize(texts).to(self.device)
            text_feats = self.model.encode_text(text_input)
            ret_vals.append(text_feats)

        if len(ret_vals) == 1:
            return ret_vals[0]

        return tuple(ret_vals)
