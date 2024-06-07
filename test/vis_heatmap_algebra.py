# %%
'''
    Script to visualize algebraic operations on heatmaps (difference, intersection).
'''
import os
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import sys
sys.path.append(os.path.realpath(os.path.join(__file__, '../../src'))) # src
from model.concept import ConceptKB, Concept
from feature_extraction import build_feature_extractor, build_sam, build_clip, build_dino
import torch
from kb_ops.train_test_split import split_from_paths
from kb_ops.build_kb import list_paths
from image_processing import build_localizer_and_segmenter
from kb_ops import ConceptKBFeaturePipeline, ConceptKBFeatureCacher
from controller import Controller
from kb_ops import CLIPConceptRetriever
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from feature_extraction.dino_features import DINOFeatureExtractor, get_rescaled_features, rescale_features
from matplotlib import colormaps
from matplotlib.gridspec import GridSpec
import logging, coloredlogs
from rembg import remove, new_session
from typing import Literal
import numpy as np
import cv2
import jsonargparse as argparse

logger = logging.getLogger(__file__)

coloredlogs.install(level='DEBUG', logger=logger)

rembg_session = new_session('isnet-general-use')

def get_foreground_mask(img: Image.Image, remove_background: bool = True):
    if remove_background:
        bw_img = remove(img, post_process_mask=True, session=rembg_session, only_mask=True) # single-channel image
        img_mask = np.array(bw_img) > 0 # (h, w)

    else:
        img_mask = np.ones((img.height, img.width), dtype=bool)

    return img_mask

def get_concept_heatmap(concept: Concept, img: Image.Image, fe: DINOFeatureExtractor):
    feature_predictor = nn.Sequential(
        concept.predictor.img_features_predictor,
        concept.predictor.img_features_weight
    ).eval().cpu()

    # Patch features
    _, patch_feats = get_rescaled_features(fe, [img], interpolate_on_cpu=True)
    patch_feats = patch_feats[0] # (resized_h, resized_w, d)
    patch_feats = rescale_features(patch_feats, img) # (h, w, d)

    # Get heatmap
    with torch.no_grad():
        # Need to move to CPU otherwise runs out of GPU mem on big images
        heatmap = feature_predictor(patch_feats.cpu()).squeeze() # (h, w)

    # Move img_features_predictor back to correct device (train is called by train method)
    feature_predictor.cuda()

    return heatmap

def get_heatmap_figure(
    concept: Concept,
    img_path: str,
    fe: DINOFeatureExtractor,
    remove_background: bool = True,
    **vis_kwargs
):
    img = Image.open(img_path).convert('RGB')
    img_mask = get_foreground_mask(img, remove_background=remove_background)

    concept_heatmap = get_concept_heatmap(concept, img, fe)

    fig = plt.figure(figsize=(15,10), constrained_layout=True)
    grid = GridSpec(2, 2, figure=fig, wspace=.05, hspace=.05)

    # Original image
    orig_img_ax = fig.add_subplot(grid[:,0])
    orig_img_ax.imshow(img)
    orig_img_ax.axis('off')
    orig_img_ax.set_title('Original Image')

    # Concept heatmap
    concept_heatmap_vis, concept_heatmap = get_heatmap_visualization(concept_heatmap, img, img_mask, **vis_kwargs)
    concept_heatmap_ax = fig.add_subplot(grid[:,1])
    concept_heatmap_ax.imshow(concept_heatmap_vis)
    concept_heatmap_ax.axis('off')
    concept_heatmap_ax.set_title(f'{concept.name.capitalize()} Heatmap')

    # Move title upwards to put distance between it and plot
    fig.suptitle('Heatmap Visualization', fontsize=24, weight='bold', y=1.05)

    return fig

def get_heatmap_differences_figure(
    concept1: Concept,
    concept2: Concept,
    img_path: str,
    fe: DINOFeatureExtractor,
    figsize=(15,10),
    intersection_min: float = .5,
    remove_background: bool = True,
    **vis_kwargs
):
    img = Image.open(img_path).convert('RGB')
    img_mask = get_foreground_mask(img, remove_background=remove_background)

    concept1_heatmap = get_concept_heatmap(concept1, img, fe)
    concept2_heatmap = get_concept_heatmap(concept2, img, fe)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = GridSpec(2, 4, figure=fig, wspace=.05, hspace=.05)

    # Original image
    orig_img_ax = fig.add_subplot(grid[:,0])
    orig_img_ax.imshow(img)
    orig_img_ax.axis('off')
    orig_img_ax.set_title('Original Image')

    # Concept 1 heatmap
    concept1_heatmap_vis, concept1_heatmap = get_heatmap_visualization(concept1_heatmap, img, img_mask, **vis_kwargs)
    concept1_heatmap_ax = fig.add_subplot(grid[0,1])
    concept1_heatmap_ax.imshow(concept1_heatmap_vis)
    concept1_heatmap_ax.axis('off')
    concept1_heatmap_ax.set_title(f'{concept1.name.capitalize()} Heatmap')

    # Concept 2 heatmap
    concept2_heatmap_vis, concept2_heatmap = get_heatmap_visualization(concept2_heatmap, img, img_mask, **vis_kwargs)
    concept2_heatmap_ax = fig.add_subplot(grid[1,1])
    concept2_heatmap_ax.imshow(concept2_heatmap_vis)
    concept2_heatmap_ax.axis('off')
    concept2_heatmap_ax.set_title(f'{concept2.name.capitalize()} Heatmap')

    # Concept1 - Concept2 heatmap
    diff_heatmap = np.clip(concept1_heatmap - concept2_heatmap, a_min=0, a_max=1)
    diff_heatmap_vis, _ = get_heatmap_visualization(diff_heatmap, img, img_mask, **vis_kwargs, is_heatmap_processed=True)
    diff_heatmap_ax = fig.add_subplot(grid[0,2])
    diff_heatmap_ax.imshow(diff_heatmap_vis)
    diff_heatmap_ax.axis('off')
    diff_heatmap_ax.set_title(f'{concept1.name.capitalize()} - {concept2.name.capitalize()} Heatmap')

    # Concept2 - Concept1 heatmap
    diff_heatmap = np.clip(concept2_heatmap - concept1_heatmap, a_min=0, a_max=1)
    diff_heatmap_vis, _ = get_heatmap_visualization(diff_heatmap, img, img_mask, **vis_kwargs, is_heatmap_processed=True)
    diff_heatmap_ax = fig.add_subplot(grid[1,2])
    diff_heatmap_ax.imshow(diff_heatmap_vis)
    diff_heatmap_ax.axis('off')
    diff_heatmap_ax.set_title(f'{concept2.name.capitalize()} - {concept1.name.capitalize()} Heatmap')

    # Concept1 and Concept2 heatmap
    intersection_mask = (concept1_heatmap > intersection_min) & (concept2_heatmap > intersection_min)
    intersection_heatmap = intersection_mask.float() * (concept1_heatmap + concept2_heatmap) / 2 # Average of heatmaps
    intersection_heatmap_vis, _ = get_heatmap_visualization(intersection_heatmap, img, img_mask, **vis_kwargs, is_heatmap_processed=True)
    intersection_heatmap_ax = fig.add_subplot(grid[:,3])
    intersection_heatmap_ax.imshow(intersection_heatmap_vis)
    intersection_heatmap_ax.axis('off')
    intersection_heatmap_ax.set_title(f'{concept1.name.capitalize()} & {concept2.name.capitalize()} Heatmap')

    # Move title upwards to put distance between it and plot
    fig.suptitle('Heatmap Differences', fontsize=24, weight='bold', y=1.05)

    return fig

def get_pre_visualization_heatmap(
    heatmap: torch.Tensor,
    img: Image.Image,
    img_mask: np.ndarray,
    strategy: Literal['normalize', 'clamp'] = 'clamp',
    is_heatmap_processed: bool = False,
    clamp_radius=5,
    clamp_center=0,
    clamp_discretize_radius=.1 # Discretization after normalizing to [0,1]
):
    '''
        Version of get_heatmap_visualization without the blending with the original image.
    '''

    logger.debug(f'Heatmap (min, max): {heatmap.min().item():.2f}, {heatmap.max().item():.2f}')
    logger.debug(f'is_heatmap_processed: {is_heatmap_processed}')

    # Mask image background
    img = np.array(img) * img_mask[..., None] # (h, w, 3)

    if strategy == 'normalize':
        if not is_heatmap_processed:
            fg_vals = heatmap[img_mask] # Foreground values
            min_val = fg_vals.min()
            max_val = fg_vals.max()

            heatmap = (heatmap - min_val) / (max_val - min_val) # Normalize

    elif strategy == 'clamp':
        if not is_heatmap_processed:
            heatmap = heatmap - clamp_center # Center at zero
            heatmap = heatmap.clamp(-clamp_radius, clamp_radius) # Restrict to [-radius, radius]
            heatmap = (heatmap + clamp_radius) / (2 * clamp_radius) # Normalize to [0, 1] with zero at .5

            heatmap[np.abs(heatmap - .5) < clamp_discretize_radius] = .5 # Discretize around .5
            heatmap[heatmap < .5 - clamp_discretize_radius] = 0
            heatmap[heatmap > .5 + clamp_discretize_radius] = 1

    return  heatmap

def get_heatmap_visualization(
    heatmap: torch.Tensor,
    img: Image.Image,
    img_mask: np.ndarray,
    strategy: Literal['normalize', 'clamp', 'hsv'] = 'clamp',
    opacity: float = .75,
    is_heatmap_processed: bool = False,
    clamp_radius=5,
    clamp_center=0,
    discretize_clamp: bool = True,
    clamp_discretize_radius=.1 # Discretization after normalizing to [0,1]
):

    logger.debug(f'Heatmap (min, max): {heatmap.min().item():.2f}, {heatmap.max().item():.2f}')
    logger.debug(f'is_heatmap_processed: {is_heatmap_processed}')

    # Mask image background
    img = np.array(img) * img_mask[..., None] # (h, w, 3)

    if strategy == 'normalize':
        if not is_heatmap_processed:
            fg_vals = heatmap[img_mask] # Foreground values
            min_val = fg_vals.min()
            max_val = fg_vals.max()

            heatmap = (heatmap - min_val) / (max_val - min_val) # Normalize

        heatmap_vis = colormaps['rainbow'](heatmap)[..., :3] # (h, w) --> (h, w, 4) --> (h, w, 3)
        heatmap_vis = opacity * heatmap_vis + (1 - opacity) * img / 255 # Blend with original image

    elif strategy == 'clamp':
        if not is_heatmap_processed:
            heatmap = heatmap - clamp_center # Center at zero
            heatmap = heatmap.clamp(-clamp_radius, clamp_radius) # Restrict to [-radius, radius]
            heatmap = (heatmap + clamp_radius) / (2 * clamp_radius) # Normalize to [0, 1] with zero at .5

            if discretize_clamp:
                heatmap[np.abs(heatmap - .5) < clamp_discretize_radius] = .5 # Discretize around .5
                heatmap[heatmap < .5 - clamp_discretize_radius] = 0
                heatmap[heatmap > .5 + clamp_discretize_radius] = 1

        heatmap_vis = colormaps["cividis"](heatmap)[
            ..., :3
        ]  # (h, w) --> (h, w, 4) --> (h, w, 3)
        # heatmap_vis = colormaps['bwr'](heatmap)[..., :3] # (h, w) --> (h, w, 4) --> (h, w, 3)
        heatmap_vis = opacity * heatmap_vis + (1 - opacity) * img / 255 # Blend with original image

    else:
        assert strategy == 'hsv'

        if not is_heatmap_processed:
            maxval  = 5
            pos = heatmap > 0
            neg = heatmap < 0
            hue = ((pos*0) + (neg*2/3))*179
            sat = (np.minimum(np.abs(heatmap/maxval),1)*255)
            hsv = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2HSV)
            val = hsv[:, :, 2]*1.0
            hsv2 = np.dstack((hue, sat, val))

        heatmap_vis = cv2.cvtColor(np.uint8(hsv2), cv2.COLOR_HSV2RGB) # This already handles blending with original image

    heatmap_vis = heatmap_vis * img_mask[..., None] # (h, w, 3); mask background

    return heatmap_vis, heatmap

def vis_checkpoint(
    ckpt_path: str,
    dir_to_vis: str = None,
    max_to_vis_per_concept: int = 3,
    remove_background: bool = True,
    infer_backtround_removal: bool = False,
    **vis_kwargs
):
    '''
        Visualize the heatmaps of existing checkpointed concept predictors on their test images.
    '''
    concept_kb = ConceptKB.load(ckpt_path)

    for concept_to_vis in concept_kb:
        # Get paths
        if dir_to_vis:
            paths = list_paths(args.dir_to_vis, exts=['.jpg', '.jpeg', '.png'])

        else:
            # Get image paths for concept
            all_paths = [
                e.image_path
                for concept in concept_kb
                for e in concept.examples
            ]

            paths = all_paths
            labels = [concept.name for concept in concept_kb for e in concept.examples]

            # (_, _), (_, _), (test_paths, test_labels) = split_from_paths(all_paths)
            # if len(test_paths) == 0:
            #     logger.warning(f'No test images for concept {concept_to_vis.name}; selecting all images including train.')
            #     test_paths = all_paths

            concept_paths = [path for i, path in enumerate(paths) if labels[i] == concept_to_vis.name][:max_to_vis_per_concept]
            remove_background_for_concept = concept_to_vis not in concept_kb.component_concepts if infer_background_removal else remove_background
            paths = concept_paths

        # Output figures
        for i, img_path in enumerate(paths, start=1):
            fig = get_heatmap_figure(concept_to_vis, img_path, dino_fe, remove_background=remove_background_for_concept, **vis_kwargs)
            fig.savefig(f'{concept_to_vis.name}_image_{i}_heatmap.jpg', bbox_inches='tight')

def vis_checkpoint_by_comparison(
    ckpt_path: str,
    concepts_to_vis: tuple[str,str],
    dir_to_vis: str = None,
    max_to_vis_per_concept: int = 3,
    remove_background: bool = True,
    **vis_kwargs
):
    '''
        Visualize the heatmaps of existing checkpointed concept predictors on each other's test images.

        If dir_to_vis is provided, visualizes the concepts on images in this directory. Else, uses each concept's test images.
    '''
    concept_kb = ConceptKB.load(ckpt_path)

    feature_pipeline = ConceptKBFeaturePipeline(loc_and_seg, feature_extractor)
    retriever = CLIPConceptRetriever(concept_kb.concepts, *clip)
    cacher = ConceptKBFeatureCacher(concept_kb, feature_pipeline)

    #  Build controller
    controller = Controller(loc_and_seg, concept_kb, feature_extractor, retriever=retriever, cacher=cacher)

    # Concepts to evaluate
    concept1_name, concept2_name = concepts_to_vis

    concept1 = controller.retrieve_concept(concept1_name)
    concept2 = controller.retrieve_concept(concept2_name)

    # Get image paths for concepts 1, 2
    all_paths = [
        e.image_path
        for concept in concept_kb
        for e in concept.examples
    ]

    # Visualize and output heatmaps
    if dir_to_vis:
        paths = list_paths(args.dir_to_vis, exts=['.jpg', '.jpeg', '.png'])

        for img_path in paths:
            fig = get_heatmap_differences_figure(concept1, concept2, img_path, dino_fe, remove_background=remove_background, **vis_kwargs)
            fig.savefig(f'{os.path.basename(img_path).split(".")[0]}_heatmaps.jpg', bbox_inches='tight')

    else: # Use test images from checkpoint
        (_, _), (_, _), (test_paths, test_labels) = split_from_paths(all_paths)

        concept1_paths = [path for i, path in enumerate(test_paths) if test_labels[i] == concept1_name][:max_to_vis_per_concept]
        concept2_paths = [path for i, path in enumerate(test_paths) if test_labels[i] == concept2_name][:max_to_vis_per_concept]

        for i, img_path in enumerate(concept1_paths, start=1):
            fig = get_heatmap_differences_figure(concept1, concept2, img_path, dino_fe, **vis_kwargs)
            fig.savefig(f'{concept1_name}_image_{i}_heatmaps.jpg', bbox_inches='tight')

        for i, img_path in enumerate(concept2_paths, start=1):
            fig = get_heatmap_differences_figure(concept2, concept1, img_path, dino_fe, **vis_kwargs) # Order just changes order of the figure, but we start with the GT here
            fig.savefig(f'{concept2_name}_image_{i}_heatmaps.jpg', bbox_inches='tight')

def parse_args(cl_args = None, config_str: str = None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_path', default='/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_03_23-19:34:45-1qycp7yt-v3-dino_pool_negatives/concept_kb_epoch_20.pt')
    parser.add_argument('--strategy', default='clamp')
    parser.add_argument('--output_dir', default='vis_checkpointed_kb')
    parser.add_argument('--max_to_vis_per_concept', default=3, type=int)
    parser.add_argument('--dir_to_vis', help='Optional path to directory of images whose heatmaps will be visualized. If not provided, will use test images from concept KB.')

    parser.add_argument('--concepts_to_compare', nargs=2, default=['mug', 'bowl'])

    parser.add_argument('--vis.strategy', default='clamp', choices=['normalize', 'clamp', 'hsv'])
    parser.add_argument('--vis.clamp_radius', default=5, type=float)
    parser.add_argument('--vis.clamp_center', default=0, type=float)
    parser.add_argument('--vis.clamp_discretize_radius', default=.1, type=float)

    if config_str:
        args = parser.parse_string(config_str)
    else:
        args = parser.parse_args(cl_args)

    return args

# %%
if __name__ == '__main__':
    # XXX While this script tries to visualize the ConceptKB's test images, depending on the split used for training the KB it may not have any test
    # images. Hence, the model may have been trained on the images themselves.

    # Original hyperparameters for XAD-v3 used clamp_center=0, clamp_radius=5., clamp_discretize_radius=.1

    # %% Build controller components
    loc_and_seg = build_localizer_and_segmenter(build_sam(), None) # Save time by not loading DesCo
    clip = build_clip()
    feature_extractor = build_feature_extractor(dino_model=build_dino(), clip_model=clip[0], clip_processor=clip[1])
    dino_fe = feature_extractor.dino_feature_extractor

    # Single-concept visualization
    output_dir = '/shared/nas2/blume5/fa23/ecole/results/vis_heatmap_sweep/single_concepts_infer_rembg'
    ckpt_path = '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_06_05-20:23:53-yd491eo3-all_planes_and_guns-infer_localize/concept_kb_epoch_26.pt'
    remove_background = False
    infer_background_removal = True
    discretize_clamp = False

    for clamp_radius in [3, 5]:
        for clamp_center in [0, 3, 6]:
            for clamp_discretize_radius in [0]: # [.1, .3, .6]:
                vis_kwargs = {
                    'clamp_radius': clamp_radius,
                    'clamp_center': clamp_center,
                    'clamp_discretize_radius': clamp_discretize_radius,
                    'discretize_clamp': discretize_clamp
                }

                sub_output_dir = os.path.join(output_dir, f'clamp_radius_{clamp_radius}_center_{clamp_center}_discretize_{discretize_clamp}_radius_{clamp_discretize_radius}')
                os.makedirs(sub_output_dir, exist_ok=True)
                os.chdir(sub_output_dir)

                vis_checkpoint(ckpt_path, remove_background=remove_background, **vis_kwargs)

    sys.exit(0)

    # Concept comparison
    concepts_to_compare_l = [
        ('biplane', 'transport plane'),
        ('cargo jet', 'passenger plane'),
        ('fighter jet', 'transport plane')
    ]

    for concepts_to_compare in concepts_to_compare_l:
        for clamp_radius in [1, 3, 5]:
            for clamp_center in [-2, -1, 0]:
                for clamp_discretize_radius in [.3, 1, 3]:

                    args = parse_args(config_str=f'''
                        ckpt_path: /shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_05_13-06:46:31-ra29szos-firerarms_more_exs_v2/concept_kb_epoch_25.pt
                        max_to_vis_per_concept: 10

                        concepts_to_compare:
                            - barrett xm109
                            - cheytac m200

                        dir_to_vis: /shared/nas2/blume5/fa23/ecole/data/firearms14k/two_subset/test
                        output_dir: /shared/nas2/blume5/fa23/ecole/vis_heatmap_sweep/clamp_radius_{clamp_radius}_center_{clamp_center}_discretize_{clamp_discretize_radius}

                        vis:
                            strategy: clamp
                            clamp_radius: {clamp_radius}
                            clamp_center: {clamp_center}
                            clamp_discretize_radius: {clamp_discretize_radius}
                    ''')

                    # Change working directory to generate outputs there
                    output_dir = os.path.realpath(args.output_dir)
                    os.makedirs(output_dir, exist_ok=True)
                    os.chdir(output_dir)

                    # Create visualizations
                    vis_checkpoint_by_comparison(args.ckpt_path, args.concepts_to_compare, max_to_vis_per_concept=args.max_to_vis_per_concept, **args.vis.as_dict())
