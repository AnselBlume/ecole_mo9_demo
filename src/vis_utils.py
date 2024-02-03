import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks
from typing import Union, List

def show(
    imgs: Union[torch.Tensor,List[torch.Tensor]],
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

            if isinstance(img, np.ndarray): # Handle numpy arrays too
                img = torch.from_numpy(img)

            img = to_pil_image(img.detach().cpu())
            ax.imshow(np.asarray(img))
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
