from PIL.Image import Image
import torch.nn as nn
from torchvision import transforms
from typing import Sequence
from tqdm import tqdm
import torch
import math
import itertools
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange, einsum, reduce
from typing import Union
from functools import partial
import logging
logger = logging.getLogger(__file__)

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

DEFAULT_RESIZE_SIZE = 256
DEFAULT_RESIZE_INTERPOLATION = transforms.InterpolationMode.BICUBIC
DEFAULT_CROP_SIZE = 224

def get_dino_transform(
    crop_img: bool,
    *,
    padding_multiple: int = 14, # aka DINOv2 model patch size
    resize_img: bool = True,
    resize_size: int = DEFAULT_RESIZE_SIZE,
    interpolation = DEFAULT_RESIZE_INTERPOLATION,
    crop_size: int = DEFAULT_CROP_SIZE,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    '''
        If crop_img is True, will automatically set resize_img to True.
    '''
    # With the default parameters, this is the transform used for DINO classification
    if crop_img:
        transforms_list = [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            _MaybeToTensor(),
            _make_normalize_transform(mean=mean, std=std),
        ]

    # Transform used for DINO segmentation in Region-Based Representations revisited and at
    # https://github.com/facebookresearch/dinov2/blob/main/notebooks/semantic_segmentation.ipynb
    else:
        transforms_list = []

        if resize_img:
            transforms_list.append(transforms.Resize(resize_size, interpolation=interpolation))

        transforms_list.extend([
            transforms.ToTensor(),
            lambda x: x.unsqueeze(0),
            CenterPadding(multiple=padding_multiple),
            transforms.Normalize(mean=mean, std=std)
        ])

    return transforms.Compose(transforms_list)

class DINOFeatureExtractor(nn.Module):
    def __init__(self, dino: nn.Module, resize_images: bool = True, crop_images: bool = False):
        super().__init__()

        self.model = dino.eval()
        self.resize_images = resize_images
        self.crop_images = crop_images
        self.transform = get_dino_transform(crop_images, resize_img=resize_images)

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

        if self.crop_images:
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

def rescale_features(
    features: torch.Tensor,
    img: Image = None,
    height: int = None,
    width: int = None,
    do_resize: bool = False,
    resize_size: Union[int, tuple[int,int]] = DEFAULT_RESIZE_SIZE
):
    '''
        Returns the features rescaled to the size of the image.

        features: (n, h_patch, w_patch, d) or (h_patch, w_patch, d)

        Returns: Interpolated features to the size of the image.
    '''
    if bool(img) + bool(width and height) + bool(do_resize) != 1:
        raise ValueError('Exactly one of img, (width and height), or do_resize must be provided')

    has_batch_dim = features.dim() > 3
    if not has_batch_dim: # Create batch dimension for interpolate
        features = features.unsqueeze(0)

    features = rearrange(features, 'n h w d -> n d h w').contiguous()

    # Resize based on min dimension or interpolate to specified dimensions
    if do_resize:
        features = TF.resize(features, resize_size, interpolation=DEFAULT_RESIZE_INTERPOLATION)

    else:
        if img:
            width, height = img.size
        features = F.interpolate(features, size=(height, width), mode='bilinear')

    features = rearrange(features, 'n d h w -> n h w d')

    if not has_batch_dim: # Squeeze the batch dimension we created to interpolate
        features = features.squeeze(0)

    return features

def get_rescaled_features(
    feature_extractor: DINOFeatureExtractor,
    images: list[Image],
    patch_size: int = 14,
    resize_size: int = DEFAULT_RESIZE_SIZE,
    crop_height: int = DEFAULT_CROP_SIZE,
    crop_width: int = DEFAULT_CROP_SIZE,
    interpolate_on_cpu: bool = False,
    fall_back_to_cpu: bool = False,
    return_on_cpu: bool = False
) -> tuple[torch.Tensor, Union[torch.Tensor, list[torch.Tensor]]]:
    '''
        Extracts features from the image and rescales them to the size of the image.

        patch_size: The patch size of the Dino model used in the DinoFeatureExtractor.
            Accessible by feature_extractor.model.patch_size.
        crop_height: The height of the cropped image, if cropping is used in the DinoFeatureExtractor.
        crop_weidth: The width of the cropped image, if cropping is used in the DinoFeatureExtractor.
        interpolate_on_cpu: If True, interpolates on CPU to avoid CUDA OOM errors.
        fall_back_to_cpu: If True, falls back to CPU if CUDA OOM error is caught.
        return_on_cpu: If True, returns the features on CPU, helping to prevent out of memory errors when storing patch features
            generated one-by-one when not resizing multiple images.

        Returns: shapes (1, d), (1, h, w, d) or list[(h, w, d) torch.Tensor]
    '''

    with torch.no_grad():
        cls_feats, patch_feats = feature_extractor(images)

    are_images_cropped = feature_extractor.crop_images
    are_images_resized = feature_extractor.resize_images

    if return_on_cpu:
        cls_feats = cls_feats.cpu()

    def patch_feats_to_cpu(patch_feats):
        if isinstance(patch_feats, torch.Tensor):
            return patch_feats.cpu()

        else:
            assert isinstance(patch_feats, list)
            assert all(isinstance(patch_feat, torch.Tensor) for patch_feat in patch_feats)

            return [
                patch_feat.cpu()
                for patch_feat in patch_feats
            ]

    def try_rescale(rescale_func, patch_feats):
        try:
            return rescale_func(patch_feats)

        except RuntimeError as e:
            if fall_back_to_cpu:
                logger.info(f'Caught out of memory error; falling back to CPU for rescaling.')
                patch_feats = patch_feats_to_cpu(patch_feats)
                return rescale_func(patch_feats)

            else:
                raise e

    # Avoid CUDA oom errors by interpolating on CPU
    if interpolate_on_cpu:
        patch_feats = patch_feats_to_cpu(patch_feats)

    # Rescale patch features
    if are_images_cropped: # All images are the same size
        rescale_func = partial(rescale_features, height=crop_height, width=crop_width)
        patch_feats = try_rescale(rescale_func, patch_feats)

        if return_on_cpu:
            patch_feats = patch_feats.cpu()

    else: # Images aren't cropped, so each patch feature has a different dimension and comes in a list
        # Rescale to padded size
        rescaled_patch_feats = []

        for patch_feat, img in zip(patch_feats, images):
            if are_images_resized: # Interpolate to padded resized size
                # Compute resized dimensions for padding removal
                resized_dims = [None, None]
                spatial_shapes = list(reversed(img.size)) # (h, w)

                smaller_dim = 0 if spatial_shapes[0] < spatial_shapes[1] else 1
                resized_dims[smaller_dim] = resize_size

                # _compute_resized_output_size floors the scaled larger dimension:
                # https://pytorch.org/vision/0.15/_modules/torchvision/transforms/functional.html
                larger_dim = 1 - smaller_dim
                resized_dims[larger_dim] = int(spatial_shapes[larger_dim] * resize_size / spatial_shapes[smaller_dim]) # Exact implementation in torch resize; multiplication first helps avoids fp errors
                # scale_factor = resize_size / spatial_shapes[smaller_dim]
                # resized_dims[larger_dim] = int(spatial_shapes[larger_dim] * scale_factor + 1e-6) # + epsilon to avoid floating point errors during floor

                height, width = resized_dims

                # Interpolate to padded resized size
                padded_resize_size = math.ceil(DEFAULT_RESIZE_SIZE / patch_size) * patch_size
                rescale_func = partial(rescale_features, do_resize=True, resize_size=padded_resize_size)

            else: # Interpolate to full, padded image size
                width, height = img.size
                padded_height = math.ceil(height / patch_size) * patch_size
                padded_width = math.ceil(width / patch_size) * patch_size
                rescale_func = partial(rescale_features, height=padded_height, width=padded_width)

            rescaled = try_rescale(rescale_func, patch_feat)

            # Remove padding from upscaled features
            rescaled = rearrange(rescaled, 'h w d -> d h w')
            rescaled = TF.center_crop(rescaled, (height, width))
            rescaled = rearrange(rescaled, 'd h w -> h w d')

            if return_on_cpu:
                rescaled = rescaled.cpu()

            rescaled_patch_feats.append(rescaled)

        patch_feats = rescaled_patch_feats

    return cls_feats, patch_feats

def interpolate_masks(
    masks: torch.BoolTensor,
    do_resize: bool = False,
    do_crop: bool = False,
    resize_size: Union[int, tuple[int, int]] = DEFAULT_RESIZE_SIZE,
    resize_interpolation = DEFAULT_RESIZE_INTERPOLATION,
    crop_size: Union[int, tuple[int, int]] = DEFAULT_CROP_SIZE
):
    masks = masks.float()

    if do_resize:
        masks = TF.resize(masks, resize_size, resize_interpolation)

    if do_crop:
        masks = TF.center_crop(masks, crop_size)

    return masks.round().bool() # Nop if not resized or cropped

def region_pool(masks: torch.BoolTensor, features: torch.Tensor):
    assert masks.shape[-2:] == features.shape[-3:-1]
    region_sums = einsum(masks.float(), features, 'n h w, n h w d -> n d') # Einsum needs floats

    # Divide by number of elements in each mask
    region_feats = region_sums / reduce(masks, 'n h w -> n', 'sum').unsqueeze(-1)

    return region_feats
