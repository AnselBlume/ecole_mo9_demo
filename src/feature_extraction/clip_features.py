from PIL.Image import Image
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class CLIPFeatureExtractor(nn.Module):
    def __init__(self, model: CLIPModel, processor: CLIPProcessor):
        super().__init__()

        self.model: CLIPModel = model.eval()
        self.processor = processor

    def forward(self, *, images: list[Image] = None, texts: list[str] = None):
        '''
            image: list[PIL.Image.Image]
            texts: list[str]
            Returns:
                If only image is provided, torch.Tensor of shape (n_imgs, enc_dim).
                If only texts is provided, torch.Tensor of shape (n_texts, enc_dim).
                If both are provided, returns a tuple of (image_feats, text_feats).
        '''
        assert images is not None or texts is not None, 'At least one of image or texts must be provided'

        # Prepare inputs
        ret_vals = []
        inputs = self.processor(image=images, text=texts, return_tensors='pt', truncation=True, padding=True)

        for k, v in inputs.items():
            inputs[k] = v.to(self.model.device)

        # Generate features
        if images is not None:
            image_feats = self.model.get_image_features(inputs['pixel_values'])
            ret_vals.append(image_feats)

        if texts is not None:
            text_feats = self.model.get_text_features(inputs['input_ids'], inputs['attention_mask'])
            ret_vals.append(text_feats)

        if len(ret_vals) == 1:
            return ret_vals[0]

        return tuple(ret_vals)
