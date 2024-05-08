# %%
'''
    Script to visualize the evolution of classifier heatmaps after training a concept one example
    at a time.
'''
import os
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys
sys.path.append(os.path.realpath(os.path.join(__file__, '../../src'))) # src
from model.concept import ConceptKB, Concept
from feature_extraction import build_feature_extractor, build_sam, build_clip, build_dino
import torch
from kb_ops.train_test_split import split_from_paths
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

CKPT_PATH = '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_03_23-19:34:45-1qycp7yt-v3-dino_pool_negatives/concept_kb_epoch_20.pt'

rembg_session = new_session('isnet-general-use')

def get_foreground_mask(img: Image.Image):
    bw_img = remove(img, post_process_mask=True, session=rembg_session, only_mask=True) # single-channel image
    img_mask = np.array(bw_img) > 0 # (h, w)

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

def get_heatmap_differences_figure(
    concept1: Concept,
    concept2: Concept,
    img_path: str,
    fe: DINOFeatureExtractor,
    figsize=(15,10),
    intersection_min: int = 0,
    **heatmap_vis_kwargs
):
    img = Image.open(img_path).convert('RGB')
    img_mask = get_foreground_mask(img)

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
    concept1_heatmap_vis = get_heatmap_visualization(concept1_heatmap, img, img_mask, **heatmap_vis_kwargs)
    concept1_heatmap_ax = fig.add_subplot(grid[0,1])
    concept1_heatmap_ax.imshow(concept1_heatmap_vis)
    concept1_heatmap_ax.axis('off')
    concept1_heatmap_ax.set_title(f'{concept1.name.capitalize()} Heatmap')

    # Concept 2 heatmap
    concept2_heatmap_vis = get_heatmap_visualization(concept2_heatmap, img, img_mask, **heatmap_vis_kwargs)
    concept2_heatmap_ax = fig.add_subplot(grid[1,1])
    concept2_heatmap_ax.imshow(concept2_heatmap_vis)
    concept2_heatmap_ax.axis('off')
    concept2_heatmap_ax.set_title(f'{concept2.name.capitalize()} Heatmap')

    # Concept1 - Concept2 heatmap
    diff_heatmap = concept1_heatmap - concept2_heatmap
    diff_heatmap_vis = get_heatmap_visualization(diff_heatmap, img, img_mask, **heatmap_vis_kwargs)
    diff_heatmap_ax = fig.add_subplot(grid[0,2])
    diff_heatmap_ax.imshow(diff_heatmap_vis)
    diff_heatmap_ax.axis('off')
    diff_heatmap_ax.set_title(f'{concept1.name.capitalize()} - {concept2.name.capitalize()} Heatmap')

    # Concept2 - Concept1 heatmap
    diff_heatmap = concept2_heatmap - concept1_heatmap
    diff_heatmap_vis = get_heatmap_visualization(diff_heatmap, img, img_mask, **heatmap_vis_kwargs)
    diff_heatmap_ax = fig.add_subplot(grid[1,2])
    diff_heatmap_ax.imshow(diff_heatmap_vis)
    diff_heatmap_ax.axis('off')
    diff_heatmap_ax.set_title(f'{concept2.name.capitalize()} - {concept1.name.capitalize()} Heatmap')

    # Concept1 and Concept2 heatmap
    intersection_mask = (concept1_heatmap > intersection_min) & (concept2_heatmap > intersection_min)
    intersection_heatmap = intersection_mask.float() * (concept1_heatmap + concept2_heatmap) / 2 # Average of heatmaps
    intersection_heatmap_vis = get_heatmap_visualization(intersection_heatmap, img, img_mask, **heatmap_vis_kwargs)
    intersection_heatmap_ax = fig.add_subplot(grid[:,3])
    intersection_heatmap_ax.imshow(intersection_heatmap_vis)
    intersection_heatmap_ax.axis('off')
    intersection_heatmap_ax.set_title(f'{concept1.name.capitalize()} & {concept2.name.capitalize()} Heatmap')

    # Move title upwards to put distance between it and plot
    fig.suptitle('Heatmap Differences', fontsize=24, weight='bold', y=1.05)

    return fig

def get_heatmap_visualization(
    heatmap: torch.Tensor,
    img: Image.Image,
    img_mask: np.ndarray,
    strategy: Literal['normalize', 'clamp', 'hsv'] = 'clamp',
    opacity: float = .75
):

    logger.debug(f'Heatmap (min, max): {heatmap.min().item():.2f}, {heatmap.max().item():.2f}')

    # Mask image background
    img = np.array(img) * img_mask[..., None] # (h, w, 3)

    if strategy == 'normalize':
        fg_vals = heatmap[img_mask] # Foreground values
        min_val = fg_vals.min()
        max_val = fg_vals.max()

        heatmap = (heatmap - min_val) / (max_val - min_val) # Normalize

        heatmap = colormaps['rainbow'](heatmap)[..., :3] # (h, w) --> (h, w, 4) --> (h, w, 3)
        heatmap = opacity * heatmap + (1 - opacity) * img / 255 # Blend with original image

    elif strategy == 'clamp':
        radius = 5
        center = 0

        heatmap = heatmap - center # Center at zero
        heatmap = heatmap.clamp(-radius, radius) # Restrict to [-radius, radius]
        heatmap = (heatmap + radius) / (2 * radius) # Normalize to [0, 1] with zero at .5

        heatmap = colormaps['bwr'](heatmap)[..., :3] # (h, w) --> (h, w, 4) --> (h, w, 3)
        heatmap = opacity * heatmap + (1 - opacity) * img / 255 # Blend with original image

    else:
        assert strategy == 'hsv'

        maxval  = 5
        pos = heatmap > 0
        neg = heatmap < 0
        hue = ((pos*0) + (neg*2/3))*179
        sat = (np.minimum(np.abs(heatmap/maxval),1)*255)
        hsv = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2HSV)
        val = hsv[:, :, 2]*1.0
        hsv2 = np.dstack((hue, sat, val))

        heatmap = cv2.cvtColor(np.uint8(hsv2), cv2.COLOR_HSV2RGB)
        # This already handles blending with original image

    heatmap = heatmap * img_mask[..., None] # (h, w, 3); mask background

    return heatmap

def vis_checkpoint(max_to_vis_per_concept: int = 3, **vis_kwargs):
    '''
        Visualize the heatmaps of existing checkpointed concept predictors on each other's test images.
    '''
    concept_kb = ConceptKB.load(CKPT_PATH)

    feature_pipeline = ConceptKBFeaturePipeline(concept_kb, loc_and_seg, feature_extractor)
    retriever = CLIPConceptRetriever(concept_kb.concepts, *clip)
    cacher = ConceptKBFeatureCacher(concept_kb, feature_pipeline)

    #  Build controller
    controller = Controller(loc_and_seg, concept_kb, feature_extractor, retriever=retriever, cacher=cacher)

    # Concepts to evaluate
    concept1_name = 'mug'
    concept2_name = 'bowl'

    concept1 = controller.retrieve_concept(concept1_name)
    concept2 = controller.retrieve_concept(concept2_name)

    # Get image paths for concepts 1, 2
    all_paths = [
        e.image_path
        for concept in concept_kb
        for e in concept.examples
    ]

    (_, _), (_, _), (test_paths, test_labels) = split_from_paths(all_paths)

    concept1_paths = [path for i, path in enumerate(test_paths) if test_labels[i] == concept1_name][:max_to_vis_per_concept]
    concept2_paths = [path for i, path in enumerate(test_paths) if test_labels[i] == concept2_name][:max_to_vis_per_concept]

    # Visualize
    vis_dir = 'vis_checkpointed_kb'
    os.makedirs(vis_dir, exist_ok=True)

    for i, img_path in enumerate(concept1_paths, start=1):
        fig = get_heatmap_differences_figure(concept1, concept2, img_path, dino_fe)
        fig.savefig(f'{vis_dir}/{concept1_name}_image_{i}_heatmaps.jpg', bbox_inches='tight')

    for i, img_path in enumerate(concept2_paths, start=1):
        fig = get_heatmap_differences_figure(concept2, concept1, img_path, dino_fe)
        fig.savefig(f'{vis_dir}/{concept2_name}_image_{i}_heatmaps.jpg', bbox_inches='tight')

def parse_args(cl_args = None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--strategy', default='clamp')
    parser.add_argument('--output_dir', default='.')

    args = parser.parse_args(cl_args)

    return args

# %%
if __name__ == '__main__':
    pass

    # %% Build controller components
    loc_and_seg = build_localizer_and_segmenter(build_sam(), None) # Save time by not loading DesCo
    clip = build_clip()
    feature_extractor = build_feature_extractor(dino_model=build_dino(), clip_model=clip[0], clip_processor=clip[1])
    dino_fe = feature_extractor.dino_feature_extractor

    args = parse_args()

    # Change working directory to generate outputs there
    output_dir = os.path.realpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    # Create visualizations
    vis_checkpoint(strategy=args.strategy)