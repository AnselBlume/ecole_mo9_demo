import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.ops
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks
from typing import Union, List, Optional
from PIL.Image import Image
import skimage 
import cv2
from pycocotools import mask as mask_utils 

def show(
    imgs: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray], Image, List[Image]],
    title: str = None,
    title_y: float = 1,
    subplot_titles: List[str] = None,
    nrows: int = 1,
    fig_kwargs: dict = {}
):
    if not isinstance(imgs, list):
        imgs = [imgs]

    ncols = int(np.ceil(len(imgs) / nrows))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, **fig_kwargs)
    fig.tight_layout()

    for i, ax in enumerate(axs.flatten()):
        if i < len(imgs):
            img = imgs[i]

            # If np.ndarray or PIL.Image, don't need to do anything, assuming ndarray is in (h,w,c)
            if isinstance(img, torch.Tensor):
                img = to_pil_image(img.detach().cpu())

            ax.imshow(img)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            # Set titles for each individual subplot
            if subplot_titles and i < len(subplot_titles):
                ax.set_title(subplot_titles[i])

        else: # Hide subplots with no images
            ax.set_visible(False)

    if title:
        fig.suptitle(title, y=title_y)

    return fig, axs

def get_colors(num_colors, cmap_name='rainbow', as_tuples=False):
    '''
    Returns a mapping from index to color (RGB).

    Args:
        num_colors (int): The number of colors to generate

    Returns:
        torch.Tensor: Mapping from index to color of shape (num_colors, 3).
    '''
    cmap = plt.get_cmap(cmap_name)

    colors = np.stack([
        (255 * np.array(cmap(i))).astype(int)[:3]
        for i in np.linspace(0, 1, num_colors)
    ])

    if as_tuples:
        colors = [tuple(c) for c in colors]

    return colors

def image_from_masks(
    masks: Union[torch.Tensor, np.ndarray],
    combine_as_binary_mask: bool = False,
    combine_color = 'aqua',
    superimpose_on_image: torch.Tensor = None,
    superimpose_alpha: float = .8,
    cmap: str = 'rainbow'
):
    '''
    Creates an image from a set of masks.

    Args:
        masks (torch.Tensor): (num_masks, height, width)
        combine_as_binary_mask (bool): Show all segmentations with the same color, showing where any mask is present. Defaults to False.
        superimpose_on_image (torch.Tensor): The image to use as the background, if provided: (C, height, width). Defaults to None.
        cmap (str, optional): Colormap name to use when coloring the masks. Defaults to 'rainbow'.

    Returns:
        torch.Tensor: Image of shape (C, height, width) with the plotted masks.
    '''
    is_numpy = isinstance(masks, np.ndarray)
    if is_numpy:
        masks = torch.from_numpy(masks)

    # Masks should be a tensor of shape (num_masks, height, width)
    if combine_as_binary_mask:
        masks = masks.sum(dim=0, keepdim=True).to(torch.bool)

    # If there is only one mask, ensure we get a visible color
    colors = get_colors(masks.shape[0], cmap_name=cmap, as_tuples=True) if masks.shape[0] > 1 else combine_color

    if superimpose_on_image is not None:
        alpha = superimpose_alpha
        background = superimpose_on_image
    else:
        alpha = 1
        background = torch.zeros(3, masks.shape[1], masks.shape[2], dtype=torch.uint8)

    masks = draw_segmentation_masks(background, masks, colors=colors, alpha=alpha)

    if is_numpy:
        masks = masks.numpy()

    return masks
def masks_to_boxes(masks:torch.Tensor):
    """
    Copy of torvision.ops.masks_to_boxes
    """
    bounding_boxes = torch.zeros((4), device=masks.device, dtype=torch.float)
    y, x = torch.where(masks[0,:,:] != 0)
    bounding_boxes[0] = torch.min(x)
    bounding_boxes[1] = torch.min(y)
    bounding_boxes[2] = torch.max(x)
    bounding_boxes[3] = torch.max(y)       

    return bounding_boxes

def mask_and_crop_image(image_file:str,mask:List):
    """
    Mask out part of image not in polygon and crop to bounding box created from mask. 

    Args:
        image_file(str): location of image to process
        mask(List): part of image to segment. Can either be a polygon (like in VAW) or RLE (output from SAM) 
    """
    image = cv2.imread(image_file)
    w,h,_ = image.shape 
    if type(mask) == list:
        segment = skimage.draw.polygon2mask((h,w),mask)
    else:
        segment = mask_utils.decode(mask)
    if segment.shape[0] == h:
        # need to transpose for cv2  
        segment = segment.T 
    segmented_image = cv2.bitwise_and(image,image,mask=segment.astype(np.uint8))
    segmented_box = masks_to_boxes(torch.from_numpy(segment).unsqueeze(0))
    x1,y1,x2,y2 = segmented_box
    segmented_image = segmented_image[int(y1):int(y2),int(x1):int(x2)]
    return segmented_image
