# %%
import os # TODO Change DesCo CUDA device here
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
    file_path = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/fork_2_9.jpg'
    img = PIL.Image.open(file_path).convert('RGB')
    img_t = pil_to_tensor(img)

    # %% Retrieve zero-shot attributes
    obj_name = os.path.splitext(os.path.basename(file_path))[0].split('_')[0]

    zs_attrs = retrieve_attributes(obj_name, controller.llm_client)
    zs_attrs = zs_attrs['required'] + zs_attrs['likely']

    # %% Get predictions
    results = controller.predict(img, zs_attrs=zs_attrs)

    part_masks = results['segmentations']['part_masks']
    attr_scores = results['scores']['part_zs_scores']['attr_probs_per_region'] # (n_attrs, n_regions)

    # %% Visualize results
    print('Original image:')
    img.show()

    print('Image regions:')
    to_pil_image(image_from_masks(part_masks, superimpose_on_image=img_t)).show()

    mask_imgs = []
    for i, attr in enumerate(zs_attrs):
        part_scores = attr_scores[i] # (n_regions,)

        # Heatmap colors from probabilities
        colors = colormap(part_scores)[:,:3] # (n_regions, 3)
        colors = [ # To list of tuples in range [0, 255]
            tuple(int(255 * c) for c in color)
            for color in colors
        ]

        # Draw masks with colors on image
        mask_img = draw_segmentation_masks(img_t, masks=part_masks, colors=colors)
        mask_imgs.append(mask_img)
        # mask_img = to_pil_image(mask_img)
        # mask_img.show()

    show(mask_imgs, subplot_titles=zs_attrs, nrows=2)

# %%
