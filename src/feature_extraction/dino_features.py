from PIL.Image import Image
import torch.nn as nn
from torchvision import transforms
from typing import Sequence
from tqdm import tqdm
import torch
import math
import itertools
import torch.nn.functional as F
from einops import rearrange

# Transforms copied from
# https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py
# and
# https://github.com/michalsr/dino_sam/blob/0742c580bcb1fb24ad2c22bb3b629f35dabd9345/extract_features.py#L96
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class CenterPadding(torch.nn.Module):
    def __init__(self, multiple = 14):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.no_grad()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output

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
    resize_img: bool,
    *,
    padding_multiple: int = 14, # aka DINOv2 model patch size
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    if resize_img:
        transforms_list = [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            _MaybeToTensor(),
            _make_normalize_transform(mean=mean, std=std),
        ]

    else:
        transforms_list = [
            transforms.ToTensor(),
            lambda x: x.unsqueeze(0),
            CenterPadding(multiple=padding_multiple),
            transforms.Normalize(mean=mean, std=std)
        ]

    return transforms.Compose(transforms_list)


class DinoFeatureExtractor(nn.Module):
    def __init__(self, dino: nn.Module, resize_images: bool = True):
        super().__init__()

        self.model = dino
        self.resize_images = resize_images
        self.transform = get_dino_transform(resize_images)

    @property
    def device(self):
        return self.model.cls_token.device

    def forward(self, images: list[Image]):
        '''
            image: list[PIL.Image.Image]
            See https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L44
            for model forward details.
        '''
        # Prepare inputs
        inputs = [self.transform(img).to(self.device) for img in images]

        if self.resize_images:
            inputs = torch.stack(inputs) # (n_imgs, 3, 224, 224)
            outputs = self.model(inputs, is_training=True) # Set is_training=True to return all outputs

            cls_tokens = outputs['x_norm_clstoken'] # (n_imgs, n_features)
            patch_tokens = outputs['x_norm_patchtokens'] # (n_imgs, n_patches, n_features)

            # Rearrange patch tokens
            n_patches_h, n_patches_w = torch.tensor(inputs.shape[-2:]) // self.model.patch_size
            patch_tokens = rearrange(patch_tokens, 'n (h w) d -> n h w d', h=n_patches_h, w=n_patches_w) # (n_imgs, n_patches_h, n_patches_w, n_features)

        else: # Padding to multiple of patch_size; need to run forward separately
            cls_tokens_l = []
            patch_tokens_l = []

            for img_t in inputs:
                outputs = self.model(img_t, is_training=True) # Set is_training=True to return all outputs

                cls_tokens = outputs['x_norm_clstoken'] # (1, n_features)
                patch_tokens = outputs['x_norm_patchtokens'] # (1, n_patches, n_features)

                # Rearrange patch tokens
                n_patches_h, n_patches_w = torch.tensor(img_t.shape[-2:]) // self.model.patch_size
                patch_tokens = rearrange(patch_tokens, '1 (h w) d -> h w d', h=n_patches_h, w=n_patches_w)

                cls_tokens_l.append(cls_tokens)
                patch_tokens_l.append(patch_tokens)

            cls_tokens = torch.cat(cls_tokens_l, dim=0) # (n_imgs, n_features)
            patch_tokens = patch_tokens_l # list[(n_patches_h, n_patches_w, n_features)]

        return cls_tokens, patch_tokens

    def forward_from_tensor(self, image: torch.Tensor):
        # Normalize & crop according to DINOv2 settings for ImageNet
        inputs = image.to(self.device)
        outputs = self.model(inputs, is_training=True) # Set is_training=True to return all outputs

        cls_token = outputs['x_norm_clstoken']
        patch_tokens = outputs['x_norm_patchtokens']

        return cls_token, patch_tokens