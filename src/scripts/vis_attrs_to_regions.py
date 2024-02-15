# %%
import os # TODO Change DesCo CUDA device here
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from controller import Controller
from model.concept import ConceptKB
from torchvision.utils import draw_segmentation_masks
from matplotlib import colormaps
import PIL
from llm.attr_retrieval import retrieve_attributes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from vis_utils import image_from_masks, show
import math
from tqdm import tqdm
import logging

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
        ConceptKB()
    )

    colormap = colormaps['rainbow']

    # %%
    in_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/graduate_descent'
    out_dir = '/shared/nas2/blume5/fa23/ecole/results/2_14_24-region_maps-only_required_with_class'
    encode_class_in_attr = True
    include_likely_attrs = False

    os.makedirs(out_dir, exist_ok=True)

    # Attributes for all classes
    all_classes = get_all_classes(in_dir)

    class_to_zs_attrs = {
        class_name : retrieve_attributes(class_name, controller.llm_client)
        for class_name in all_classes
    }

    class_to_zs_attrs = { # Extract attributes from dictionary
        k : v['required'] + (v['likely'] if include_likely_attrs else [])
        for k, v in class_to_zs_attrs.items()
    }

    if encode_class_in_attr:
        class_to_zs_attrs = { # Encode class name to disambiguate attribute/class collisions
            class_name : [
                f'{attr} of {controller._get_article(class_name)}{class_name}'
                for attr in attrs
            ]
            for class_name, attrs in class_to_zs_attrs.items()
        }

    # Generate results
    paths = sorted(os.listdir(in_dir))
    prog_bar = tqdm(paths)

    for basename in prog_bar:
        prog_bar.set_description(basename)
        file_name, ext = os.path.splitext(basename)
        if ext not in ['.jpg', '.png']:
            continue

        file_path = os.path.join(in_dir, basename)

        img = PIL.Image.open(file_path).convert('RGB')
        img_t = pil_to_tensor(img)

        # Retrieve zero-shot attributes
        obj_name = get_class_name(basename)
        zs_attrs = class_to_zs_attrs[obj_name]

        # Get predictions
        results = controller.predict(img, zs_attrs=zs_attrs)

        part_masks = results['segmentations']['part_masks']
        attr_scores = results['scores']['part_zs_scores']['attr_probs_per_region'] # (n_attrs, n_regions)

        if len(part_masks) == 0:
            logger.warning(f'No regions found for {file_path}')
            out_path = os.path.join(out_dir, f'{file_name}_no_regions_after_filtering.png')

            with open(out_path, 'w') as f: # Create empty file
                pass

            continue

        # Visualize results
        img.save(os.path.join(out_dir, f'{file_name}_original.png'))

        image_regions = to_pil_image(image_from_masks(part_masks, superimpose_on_image=img_t))
        image_regions.save(os.path.join(out_dir, f'{file_name}_regions.png'))

        heatmap_imgs = []
        maximizing_region_imgs = []
        for i, attr in enumerate(zs_attrs):
            part_scores = attr_scores[i] # (n_regions,)
            maximizing_region = part_scores.argmax().item()

            # Heatmap colors from probabilities
            colors = colormap(part_scores)[:,:3] # (n_regions, 3)
            colors = [ # To list of tuples in range [0, 255]
                tuple(int(255 * c) for c in color)
                for color in colors
            ]

            # Draw masks with colors on image
            heatmap_img = draw_segmentation_masks(img_t, masks=part_masks, colors=colors)
            heatmap_imgs.append(heatmap_img)

            # Draw mask of maximizing region
            maximizing_region_img = image_from_masks(part_masks[maximizing_region][None,...], combine_as_binary_mask=True, superimpose_on_image=img_t)
            maximizing_region_imgs.append(maximizing_region_img)

        # Show results
        nrows = math.ceil(len(heatmap_imgs) / 3)

        fig, ax = show(heatmap_imgs, subplot_titles=zs_attrs, nrows=nrows, title='Region Heatmaps')
        fig.savefig(os.path.join(out_dir, f'{file_name}_region_heatmaps.png'))

        fig, ax = show(maximizing_region_imgs, subplot_titles=zs_attrs, nrows=nrows, title='Maximizing regions')
        fig.savefig(os.path.join(out_dir, f'{file_name}_maximizing_regions.png'))

    # %%
