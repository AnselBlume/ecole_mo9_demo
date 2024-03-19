from PIL.Image import Image
import torch.nn as nn
from torchvision import transforms
from typing import Sequence
from tqdm import tqdm
import torch

# Copied from https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class _MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)

def _make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)

def get_dino_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        _MaybeToTensor(),
        _make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)


class DinoFeatureExtractor(nn.Module):
    def __init__(self, dino: nn.Module):
        super().__init__()

        self.model = dino
        self.transform = get_dino_transform()
        self.device = self.model.cls_token.device

    def forward(self, images: list[Image]):
        '''
            image: list[PIL.Image.Image]
            See https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L44
            for model forward details.
        '''
        # Prepare inputs
        inputs = torch.stack([self.transform(img) for img in images]).to(self.device)
        outputs = self.model(inputs, is_training=True) # Set is_training=True to return all outputs

        cls_token = outputs['x_norm_clstoken']
        patch_tokens = outputs['x_norm_patchtokens']

        return cls_token, patch_tokens

    def forward_from_tensor(self, image: torch.Tensor):
        # Normalize & crop according to DINOv2 settings for ImageNet
        inputs = image.to(self.device)
        outputs = self.model(inputs, is_training=True) # Set is_training=True to return all outputs

        cls_token = outputs['x_norm_clstoken']
        patch_tokens = outputs['x_norm_patchtokens']

        return cls_token, patch_tokens