# %%
import os # TODO Change DesCo CUDA device here
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from controller import Controller
from model.concept import ConceptDB
from torchvision.utils import draw_segmentation_masks
from matplotlib import colormaps
import PIL
from llm.attr_retrieval import retrieve_attributes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from vis_utils import image_from_masks, show
import torch
import math
from tqdm import tqdm
import logging
from itertools import chain
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def get_class_name(path: str):
    return os.path.splitext(os.path.basename(path))[0].split('_')[0].lower()

def get_all_classes(in_dir: str):
    paths = sorted(os.listdir(in_dir))
    classes = set()

    for basename in paths:
        file_name, ext = os.path.splitext(basename)
        if ext not in ['.jpg', '.png']:
            continue

        obj_name = get_class_name(basename)
        classes.add(obj_name)

    return sorted(classes)

def get_class_scores(all_zs_attr_scores: torch.Tensor, class_to_zs_attrs: dict, controller: Controller):
    '''
    Args:
        all_zs_attr_scores: torch.Tensor of shape (n_all_zs_attrs, n_regions)
        class_to_zs_attrs: dict[str, List[str]] of class name to list of zero-shot attributes
        controller: Controller

    Returns:
        class_scores: torch.Tensor of shape (n_classes,)
    '''
    all_zs_attr_scores = all_zs_attr_scores.split( # List[torch.Tensor] of shape (n_class_zs_attrs, n_regions)
        [len(l) for l in class_to_zs_attrs.values()]
    , dim=1)

    class_scores = torch.tensor([ # Normalize by n_images, n_attrs and sum
        controller.attr_scorer._apply_weights(s).sum()
        for s in all_zs_attr_scores
    ]).softmax(dim=0)

    return class_scores

# %%
if __name__ == '__main__':
    # Import here so DesCo sees the CUDA device change
    from predictors import (
        build_clip,
        build_zero_shot_attr_predictor,
        build_desco,
        build_sam,
    )

    controller = Controller(
        build_sam(),
        build_desco(),
        build_zero_shot_attr_predictor(*build_clip()),
        ConceptDB()
    )

    colormap = colormaps['rainbow']

    # %%
    in_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/graduate_descent'
    out_dir = '/shared/nas2/blume5/fa23/ecole/results/2_14_24-region_maps-only_required'

    os.makedirs(out_dir, exist_ok=True)

    # Attributes for all classes
    all_classes = get_all_classes(in_dir)

    class_to_zs_attrs = {
        class_name : retrieve_attributes(class_name, controller.llm_client)
        for class_name in all_classes
    }

    class_to_zs_attrs = { # Extract attributes from dictionary
        k : v['required'] # + v['likely']
        for k, v in class_to_zs_attrs.items()
    }

    class_to_zs_attrs = { # Encode class name to disambiguate attribute/class collisions
        class_name : [
            f'{attr} of {controller._get_article(class_name)}{class_name}'
            for attr in attrs
        ]
        for class_name, attrs in class_to_zs_attrs.items()
    }

    all_zs_attrs = list(chain.from_iterable(class_to_zs_attrs.values())) # list[str]

    # Generate results
    paths = sorted(os.listdir(in_dir))
    prog_bar = tqdm(paths)

    # %%
    for basename in prog_bar:
        prog_bar.set_description(basename)
        file_name, ext = os.path.splitext(basename)
        if ext not in ['.jpg', '.png']:
            continue

        file_path = os.path.join(in_dir, basename)

        img = PIL.Image.open(file_path).convert('RGB')
        img_t = pil_to_tensor(img)

        # Retrieve zero-shot attributes
        class_name = get_class_name(basename)

        # Get predictions
        results = controller.predict(img, zs_attrs=all_zs_attrs)

        part_attr_scores = results['scores']['part_zs_scores']['raw_scores'] # (n_all_zs_attrs, n_regions)
        part_class_scores = get_class_scores(part_attr_scores, class_to_zs_attrs, controller) # (n_classes,)

        full_attr_scores = results['scores']['full_zs_scores']['raw_scores'] # (n_all_zs_attrs,)
        full_class_scores = get_class_scores(full_attr_scores, class_to_zs_attrs, controller) # (n_classes,)

        # Visualize regions
        part_masks = results['segmentations']['part_masks']

        if len(part_masks) == 0:
            part_masks = torch.ones(1, *img_t.shape[-2:], dtype=torch.bool)

        regions_img = image_from_masks(part_masks, combine_as_binary_mask=len(part_masks) == 0, superimpose_on_image=img_t)

        # Plot scores
        fig, axs = plt.subplots(nrows=2, ncols=2)

        axs[0][0].imshow(to_pil_image(regions_img))
        # Set title to bottom of plot
        axs[0][0].set_title('Regions')
        axs[0][0].axis('off')

        y_max = axs[0][1].get_ylim()[1]
        class_attrs = class_to_zs_attrs[class_name]
        axs[0][1].set_title('Class Attributes')
        attr_str = '\n'.join([f'- {attr}' for attr in class_attrs])
        axs[0][1].text(0, y_max - .1*y_max, attr_str, fontsize=8, verticalalignment='top')
        # for i, attr in enumerate(class_attrs):
        #     axs[0][1].text(0, y_max - y_max / len(class_attrs) * i, f'- {attr}, fontsize=8, verticalalignment='top')
        axs[0][1].axis('off')

        axs[1][0].barh(all_classes, part_class_scores)
        axs[1][0].set_title('Part scores')

        axs[1][1].barh(all_classes, full_class_scores)
        axs[1][1].set_title('Full scores')

        fig.suptitle(file_name)
        fig.tight_layout()

        break

        # # Visualize results
        # print('Original image:')
        # # img.show()
        # img.save(os.path.join(out_dir, f'{file_name}_original.png'))

        # print('Image regions:')
        # image_regions = to_pil_image(image_from_masks(part_masks, superimpose_on_image=img_t))
        # # image_regions.show()
        # image_regions.save(os.path.join(out_dir, f'{file_name}_regions.png'))

        # heatmap_imgs = []
        # maximizing_region_imgs = []
        # for i, attr in enumerate(zs_attrs):
        #     part_scores = attr_scores[i] # (n_regions,)
        #     maximizing_region = part_scores.argmax().item()

        #     # Heatmap colors from probabilities
        #     colors = colormap(part_scores)[:,:3] # (n_regions, 3)
        #     colors = [ # To list of tuples in range [0, 255]
        #         tuple(int(255 * c) for c in color)
        #         for color in colors
        #     ]

        #     # Draw masks with colors on image
        #     heatmap_img = draw_segmentation_masks(img_t, masks=part_masks, colors=colors)
        #     heatmap_imgs.append(heatmap_img)

        #     # Draw mask of maximizing region
        #     maximizing_region_img = image_from_masks(part_masks[maximizing_region][None,...], combine_as_binary_mask=True, superimpose_on_image=img_t)
        #     maximizing_region_imgs.append(maximizing_region_img)

        # # Show results
        # nrows = math.ceil(len(heatmap_imgs) / 3)

        # # fig, ax = show(heatmap_imgs, subplot_titles=zs_attrs, nrows=nrows, title='Region Heatmaps')
        # # fig.savefig(os.path.join(out_dir, f'{file_name}_region_heatmaps.png'))

        # fig, ax = show(maximizing_region_imgs, subplot_titles=zs_attrs, nrows=nrows, title='Maximizing regions')
        # fig.savefig(os.path.join(out_dir, f'{file_name}_maximizing_regions.png'))

    # %%
