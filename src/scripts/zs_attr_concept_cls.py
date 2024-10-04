# %%
import os  # Change DesCo CUDA device here

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import logging
import math
from itertools import chain

import matplotlib.pyplot as plt
import PIL
import torch
from controller import Controller
from llm.attr_retrieval import retrieve_attributes
from matplotlib.gridspec import GridSpec
from model.concept import ConceptKB
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from tqdm import tqdm
from visualization.vis_utils import image_from_masks

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
    from feature_extraction import (build_clip, build_desco, build_sam,
                                    build_zero_shot_attr_predictor)
    from image_processing import build_localizer_and_segmenter

    controller = Controller(
        build_localizer_and_segmenter(build_sam(), build_desco()),
        ConceptKB(),
        zs_predictor=build_zero_shot_attr_predictor(*build_clip())
    )

    # %%
    in_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/graduate_descent'
    out_dir = '/shared/nas2/blume5/fa23/ecole/results/2_14_24-predictions-class_names'
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

    all_zs_attrs = list(chain.from_iterable(class_to_zs_attrs.values())) # list[str]

    # Generate results
    paths = sorted(os.listdir(in_dir))
    prog_bar = tqdm(paths)

    accuracies = {'parts': 0, 'full': 0}
    total_count = 0

    # %%
    for basename in prog_bar:
        prog_bar.set_description(basename)
        file_name, ext = os.path.splitext(basename)
        if ext not in ['.jpg', '.png']:
            continue

        total_count += 1

        file_path = os.path.join(in_dir, basename)

        img = PIL.Image.open(file_path).convert('RGB')
        img_t = pil_to_tensor(img)

        # Retrieve zero-shot attributes
        class_name = get_class_name(basename)

        # Get predictions
        results = controller.predict_from_zs_attributes(img, zs_attrs=all_zs_attrs)

        part_attr_scores = results['scores']['part_zs_scores']['raw_scores'] # (n_all_zs_attrs, n_regions)
        part_class_scores = get_class_scores(part_attr_scores, class_to_zs_attrs, controller) # (n_classes,)

        if part_class_scores.argmax() == all_classes.index(class_name): # Update accuracy
            accuracies['parts'] += 1

        full_attr_scores = results['scores']['full_zs_scores']['raw_scores'] # (n_all_zs_attrs,)
        full_class_scores = get_class_scores(full_attr_scores, class_to_zs_attrs, controller) # (n_classes,)

        if full_class_scores.argmax() == all_classes.index(class_name): # Update accuracy
            accuracies['full'] += 1

        # Visualize regions
        part_masks = results['segmentations'].part_masks

        if len(part_masks) == 0:
            part_masks = torch.ones(1, *img_t.shape[-2:], dtype=torch.bool)

        regions_img = image_from_masks(part_masks, combine_as_binary_mask=len(part_masks) == 0, superimpose_on_image=img_t)

        # Plot scores
        # fig, axs = plt.subplots(nrows=2, ncols=2)

        fig = plt.figure()
        gs = GridSpec(nrows=2, ncols=6, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.imshow(img)
        ax1.set_title('Original Image')
        ax1.axis('off')

        # Regions
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax2.imshow(to_pil_image(regions_img))
        ax2.set_title('Regions')
        ax2.axis('off')

        # axs[0][0].imshow(to_pil_image(regions_img))
        # axs[0][0].set_title('Regions')
        # axs[0][0].axis('off')

        # Class attributes
        ax3 = fig.add_subplot(gs[0, 4:6])

        class_attrs = class_to_zs_attrs[class_name]
        attr_str = '\n'.join([f'- {attr}' for attr in class_attrs])

        y_max = ax3.get_ylim()[1]
        ax3.text(0, y_max - .1*y_max, attr_str, fontsize=8, verticalalignment='top')
        ax3.set_title('Class Attributes', loc='left')
        ax3.axis('off')

        # y_max = axs[0][1].get_ylim()[1]
        # axs[0][1].set_title('Class Attributes')
        # attr_str = '\n'.join([f'- {attr}' for attr in class_attrs])
        # axs[0][1].text(0, y_max - .1*y_max, attr_str, fontsize=8, verticalalignment='top')
        # axs[0][1].axis('off')

        # Part scores
        ax4 = fig.add_subplot(gs[1, 0:3])
        ax4.barh(all_classes, part_class_scores)
        ax4.set_title('Part scores')

        # axs[1][0].barh(all_classes, part_class_scores)
        # axs[1][0].set_title('Part scores')

        # Full image scores
        ax5 = fig.add_subplot(gs[1, 3:6])
        ax5.barh(all_classes, full_class_scores)
        ax5.set_title('Full scores')

        # axs[1][1].barh(all_classes, full_class_scores)
        # axs[1][1].set_title('Full scores')

        fig.suptitle(file_name)
        fig.tight_layout()

        # fig.show()
        fig.savefig(os.path.join(out_dir, f'{file_name}.png'))

    # Dump accuracy
    accuracies['parts'] /= total_count
    accuracies['full'] /= total_count

    out_file = os.path.join(out_dir, 'accuracies.json')

    with open(out_file, 'w') as f:
        json.dump(accuracies, f, indent=4)
