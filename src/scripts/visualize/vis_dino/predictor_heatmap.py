# %%
'''
    Script to visualize the evolution of classifier heatmaps after training a concept one example
    at a time.
'''
import os
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys
sys.path.append(os.path.realpath(os.path.join(__file__, '../../..'))) # src
sys.path.append(os.path.realpath(os.path.join(__file__, '../../../../test'))) # For test.train_from_scratch
from model.concept import ConceptKB, ConceptExample, ConceptKBConfig, Concept
from feature_extraction import build_feature_extractor, build_sam, build_desco, build_clip, build_dino
import torch
from kb_ops.train_test_split import split_from_paths
from train_from_scratch import prepare_concept # in test module
from image_processing import build_localizer_and_segmenter
from kb_ops import ConceptKBFeaturePipeline, ConceptKBFeatureCacher
from feature_extraction.trained_attrs import N_ATTRS_DINO
from controller import Controller
from kb_ops import CLIPConceptRetriever
from scripts.utils import set_feature_paths
from kb_ops.build_kb import list_paths
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from feature_extraction.dino_features import DINOFeatureExtractor, get_rescaled_features, rescale_features
from matplotlib import colormaps
import logging, coloredlogs
from rembg import remove, new_session
from typing import Literal
import numpy as np
import cv2
import jsonargparse as argparse

logger = logging.getLogger(__file__)

coloredlogs.install(level='DEBUG', logger=logger)

CKPT_PATH = '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_03_23-19:34:45-1qycp7yt-v3-dino_pool_negatives/concept_kb_epoch_20.pt'

def vis_concept_predictor_heatmap(
    concept: Concept,
    img_path: str,
    fe: DINOFeatureExtractor,
    figsize=(15, 10),
    strategy: Literal['normalize', 'clamp', 'hsv'] = 'clamp',
    opacity: float = .75,
    title=None
):
    # Image and image predictor
    img = Image.open(img_path).convert('RGB')

    bw_img = remove(img, post_process_mask=True, session=new_session('isnet-general-use'), only_mask=True) # single-channel image
    img_mask = np.array(bw_img) > 0 # (h, w)

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

    logger.debug(f'Heatmap (min, max): {heatmap.min().item():.2f}, {heatmap.max().item():.2f}')

    fg_vals = heatmap[img_mask] # Foreground values
    cum_score = fg_vals.sum() # Total score across foreground

    # Move img_features_predictor back to correct device (train is called by train method)
    feature_predictor.cuda()

    # Visualize original image
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # Mask image background
    img = np.array(img) * img_mask[..., None] # (h, w, 3)

    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    if strategy == 'normalize':
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

    axs[1].imshow(heatmap)
    axs[1].set_title(f'Predictor Heatmap (Feature Score: {cum_score * 100:.2f})')
    axs[1].axis('off')

    if title:
        fig.suptitle(title)

    return fig, axs

def vis_evolution(**vis_kwargs):
    '''
        Visualize the evolution of concept predictor heatmaps after training a new concept one example at a time.
        Both the original and the new concept have limited training data.

        Takes a new concept's images, and trains its concept predictor (and an existing similar one) iteratively,
        adding one image at a time to the train set.
    '''
    concept_kb = ConceptKB()
    concept_kb.initialize(ConceptKBConfig(
        n_trained_attrs=N_ATTRS_DINO,
    ))

    feature_pipeline = ConceptKBFeaturePipeline(concept_kb, loc_and_seg, feature_extractor)
    retriever = CLIPConceptRetriever(concept_kb.concepts, *clip)
    cacher = ConceptKBFeatureCacher(concept_kb, feature_pipeline)

    # %% Build controller
    controller = Controller(loc_and_seg, concept_kb, feature_extractor, retriever=retriever, cacher=cacher)

    # %% Add first concept and train
    img_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/single_concepts/hoes'
    concept1_name = 'hoe'
    cache_dir = '/shared/nas2/blume5/fa23/ecole/cache/hoes'

    prepare_concept(img_dir, concept1_name, cache_dir, controller, set_segmentation_paths=True)
    concept1 = controller.retrieve_concept(concept1_name)
    controller.train_concept(concept1_name)

    # %% Add second concept
    img_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/single_concepts/shovels'
    concept2_name = 'shovel'
    cache_dir = '/shared/nas2/blume5/fa23/ecole/cache/shovels'

    img_paths = prepare_concept(img_dir, concept2_name, cache_dir, controller, set_segmentation_paths=True)

    # %% Train second concept, visualizing each
    vis_dir = 'vis_evolution'
    os.makedirs(vis_dir, exist_ok=True)

    controller.train_concept(concept2_name)
    img_paths = list_paths(img_dir, exts=['.jpg', '.png'])
    concept2 = controller.add_concept(concept2_name)

    for i, img_path in enumerate(img_paths, start=1):
        new_example = ConceptExample(image_path=img_path)
        concept2.examples.append(new_example)
        set_feature_paths([concept2], segmentations_dir=cache_dir + '/segmentations')

        # %% Visualize, train, and visualize concept 2
        fig, axs = vis_concept_predictor_heatmap(
            concept2,
            img_path,
            dino_fe,
            title=f'{concept2_name.capitalize()} Predictor Heatmap Before Image {i}',
            **vis_kwargs
        )
        fig.savefig(f'{vis_dir}/{concept2_name}_heatmap_before_image_{i}.jpg')

        controller.train_concept(concept2_name) # Train on all existing images up to this point, like concept 1

        fig, axs = vis_concept_predictor_heatmap(
            concept2,
            img_path,
            dino_fe,
            title=f'{concept2_name.capitalize()} Predictor Heatmap After Image {i}',
            **vis_kwargs
        )
        fig.savefig(f'{vis_dir}/{concept2_name}_heatmap_after_image_{i}.jpg')

        # %% Visualize, train, and visualize concept 1
        fig, axs = vis_concept_predictor_heatmap(
            concept1,
            img_path,
            dino_fe,
            title=f'{concept1_name.capitalize()} Predictor Heatmap Before Image {i}',
            **vis_kwargs
        )
        fig.savefig(f'{vis_dir}/{concept1_name}_heatmap_before_image_{i}')

        controller.train_concept(concept1_name, stopping_condition='n_epochs', n_epochs=5)

        fig, axs = vis_concept_predictor_heatmap(
            concept1,
            img_path,
            dino_fe,
            title=f'{concept1_name.capitalize()} Predictor Heatmap After Image {i}',
            **vis_kwargs
        )
        fig.savefig(f'{vis_dir}/{concept1_name}_heatmap_after_image_{i}.jpg')

def vis_checkpoint_new_concept(max_to_vis_per_concept: int = 3, **vis_kwargs):
    '''
        Visualize the heatmaps of the concept predictors after adding a new concept to the ConceptKB.
        First trains the new concept, then retrains an old similar concept on the new data for comparison.
    '''
    vis_dir = 'vis_checkpointed_kb_new_concept'

    os.makedirs(vis_dir, exist_ok=True)
    concept_kb = ConceptKB.load(CKPT_PATH)

    feature_pipeline = ConceptKBFeaturePipeline(concept_kb, loc_and_seg, feature_extractor)
    retriever = CLIPConceptRetriever(concept_kb.concepts, *clip)
    cacher = ConceptKBFeatureCacher(concept_kb, feature_pipeline)

    # %% Build controller
    controller = Controller(loc_and_seg, concept_kb, feature_extractor, retriever=retriever, cacher=cacher)

    # %% Add new concept (hoe/concept1 is already added)
    img_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/single_concepts/shovels'
    concept2_name = 'shovel'
    cache_dir = '/shared/nas2/blume5/fa23/ecole/cache/shovels'

    prepare_concept(img_dir, concept2_name, cache_dir, controller, set_segmentation_paths=True)
    concept2 = controller.retrieve_concept(concept2_name)
    controller.train_concept(concept2_name)

    # Visualize concept 1 before training on concept 2 examples
    concept1_name = 'hoe' # For comparison to concept 2
    concept1 = controller.retrieve_concept(concept1_name)

    for i, img_path in enumerate([e.image_path for e in concept2.examples], start=1):
        fig, axs = vis_concept_predictor_heatmap(concept1, img_path, dino_fe, title=f'Before Retraining {concept1_name.capitalize()} Predictor Heatmap', **vis_kwargs)
        fig.savefig(f'{vis_dir}/image_{i}_before_retraining_{concept1_name}_heatmap.jpg')

    # Retrain concept one
    controller.train_concept(concept1_name, stopping_condition='n_epochs', n_epochs=10)

    # For each image in each concept, visualize the heatmaps of both predictors
    for concept in [concept1, concept2]:
        img_paths = [e.image_path for e in concept.examples][:max_to_vis_per_concept]

        for i, img_path in enumerate(img_paths, start=1):
            # Visualize predictions from both predictors
            fig, axs = vis_concept_predictor_heatmap(concept1, img_path, dino_fe, title=f'{concept1.name.capitalize()} Predictor Heatmap', **vis_kwargs)
            fig.savefig(f'{vis_dir}/image_{i}_after_retraining_{concept1_name}_heatmap.jpg')

            fig, axs = vis_concept_predictor_heatmap(concept2, img_path, dino_fe, title=f'{concept2.name.capitalize()} Predictor Heatmap', **vis_kwargs)
            fig.savefig(f'{vis_dir}/image_{i}_{concept2_name}_heatmap.jpg')

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
    concept1_name = 'bowl'
    concept2_name = 'mug'

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
        fig, axs = vis_concept_predictor_heatmap(concept1, img_path, dino_fe, title=f'{concept1_name.capitalize()} Predictor Heatmap', **vis_kwargs)
        fig.savefig(f'{vis_dir}/{concept1_name}_image_{i}_{concept1_name}_heatmap.jpg')

        fig, axs = vis_concept_predictor_heatmap(concept2, img_path, dino_fe, title=f'{concept2_name.capitalize()} Predictor Heatmap', **vis_kwargs)
        fig.savefig(f'{vis_dir}/{concept1_name}_image_{i}_{concept2_name}_heatmap.jpg')

    for i, img_path in enumerate(concept2_paths, start=1):
        fig, axs = vis_concept_predictor_heatmap(concept1, img_path, dino_fe, title=f'{concept1_name.capitalize()} Predictor Heatmap', **vis_kwargs)
        fig.savefig(f'{vis_dir}/{concept2_name}_image_{i}_{concept1_name}_heatmap.jpg')

        fig, axs = vis_concept_predictor_heatmap(concept2, img_path, dino_fe, title=f'{concept2_name.capitalize()} Predictor Heatmap', **vis_kwargs)
        fig.savefig(f'{vis_dir}/{concept2_name}_image_{i}_{concept2_name}_heatmap.jpg')

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
    vis_evolution(strategy=args.strategy)
    vis_checkpoint_new_concept(strategy=args.strategy)
    vis_checkpoint(strategy=args.strategy)