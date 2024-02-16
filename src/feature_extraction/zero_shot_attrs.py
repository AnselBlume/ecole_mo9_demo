# %%
if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from typing import Iterable
from transformers import CLIPModel, CLIPProcessor
from PIL.Image import Image
import torch
from typing import Optional
import torch.linalg as LA

class CLIPAttributePredictor:
    def __init__(self, clip: CLIPModel, processor: CLIPProcessor):
        self.clip = clip.eval()
        self.device = next(clip.parameters()).device
        self.processor = processor

    @torch.no_grad()
    def predict(self, images: Iterable[Image], texts: list[str], apply_sigmoid: bool = False, threshold: Optional[float] = None):
        '''
            Returns a torch.Tensor of matching scores in [0, 1] with shape (1, num_texts) if threshold is None.
            Otherwise, returns a torch.BoolTensor with shape (1, num_texts) indicating values above threshold.
        '''
        # Prepare model inputs
        preprocessed = self.processor(images=images, text=texts, return_tensors='pt', padding=True, truncation=True)

        for k, v in preprocessed.items():
            preprocessed[k] = v.to(self.device)

        # Get scores
        outputs = self.clip(**preprocessed)
        preds = outputs.logits_per_image

        # Potentially apply sigmoid and threshold
        if apply_sigmoid:
            preds = preds.sigmoid()

        if threshold is not None:
            assert apply_sigmoid, 'Thresholding only supported when apply_sigmoid is True'
            assert 0 <= threshold <= 1, 'Threshold must be between 0 and 1'
            preds = preds > threshold

        return preds

    @torch.no_grad()
    def feature_score(self, img_feats: torch.Tensor, text_feats: torch.Tensor):
        '''
            Args:
                img_feats: torch.Tensor of shape (n_imgs, d)
                text_feats: torch.Tensor of shape (n_texts, d)

            Returns:
                torch.Tensor of shape (n_imgs, n_texts) with matching scores in [-1, 1]
        '''
        img_feats = img_feats / LA.norm(img_feats, dim=-1, keepdim=True)
        text_feats = text_feats / LA.norm(text_feats, dim=-1, keepdim=True)

        return torch.matmul(img_feats, text_feats.T)

# %%
if __name__ == '__main__':
    # Example usage
    import PIL
    import sys
    from feature_extraction import build_clip

    predictor = CLIPAttributePredictor(*build_clip(device='cuda'))

    # %%
    image = PIL.Image.open('/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/dog.png').convert('RGB')

    texts = ['a dog', 'a dog and a bowl', 'a cat', 'a mug', 'orange']

    scores = predictor.predict(image, texts)
    print({
        text: score.item()
        for text, score in zip(texts, scores[0])
    })
# %%
