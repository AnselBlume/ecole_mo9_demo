# %%
if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from clip.model import CLIP
from torchvision.transforms import Compose
from PIL.Image import Image
import torch
from typing import Optional
import clip

class CLIPAttributePredictor:
    def __init__(self, clip: CLIP, preprocess: Compose):
        self.clip = clip.eval()
        self.device = next(clip.parameters()).device
        self.preprocess = preprocess

    @torch.inference_mode()
    def predict(self, image: Image, texts: list[str], apply_sigmoid: bool = False, threshold: Optional[float] = None):
        '''
            Returns a torch.Tensor of matching scores in [0, 1] with shape (1, num_texts) if threshold is None.
            Otherwise, returns a torch.BoolTensor with shape (1, num_texts) indicating values above threshold.
        '''
        # Compute scores
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        texts = clip.tokenize(texts).to(self.device)
        logits_per_image, logits_per_text = self.clip(image, texts)

        preds = logits_per_image

        # Potentially apply sigmoid and threshold
        if apply_sigmoid:
            preds = preds.sigmoid()

        if threshold is not None:
            assert apply_sigmoid, 'Thresholding only supported when apply_sigmoid is True'
            assert 0 <= threshold <= 1, 'Threshold must be between 0 and 1'
            preds = logits_per_image > threshold

        return preds

# %%
if __name__ == '__main__':
    # Example usage
    import PIL
    import sys
    from predictors import build_clip

    predictor = CLIPAttributePredictor(*build_clip())

    # %%
    image = PIL.Image.open('/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/dog.png')

    texts = ['a dog', 'a dog and a bowl', 'a cat', 'a mug', 'orange']

    scores = predictor.predict(image, texts)
    print({
        text: score.item()
        for text, score in zip(texts, scores[0])
    })
# %%
