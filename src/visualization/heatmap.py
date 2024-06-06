# %%
'''
    Based on test/vis_heatmap_algebra.py.
    Not all functions from that file (including intersection/and) are implemented here.
    See that file for more and for hyperparameter sweeps.
'''
if __name__ == '__main__':
    import os
    import sys
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    sys.path.append('/shared/nas2/blume5/fa23/ecole/src/mo9_demo/src')

import rembg.sessions
from model.concept import ConceptKB, Concept
from dataclasses import dataclass
from PIL import Image
from enum import Enum
import torch
import torch.nn as nn
import rembg
from rembg import remove, new_session
from feature_extraction.dino_features import DINOFeatureExtractor, get_rescaled_features, rescale_features
from matplotlib import colormaps
import numpy as np
from matplotlib.gridspec import GridSpec
import cv2
import logging

logger = logging.getLogger(__file__)

class HeatmapStrategy(Enum):
    NORMALIZE = 'normalize'
    CLAMP = 'clamp'
    HSV = 'hsv'

@dataclass
class HeatmapVisualizerConfig:
    strategy: HeatmapStrategy = HeatmapStrategy.CLAMP

    opacity: float = .75

    remove_background: bool = True

    # Clamp settings
    clamp_radius: int = 5
    clamp_center: int = 6
    clamp_discretize_bounds: tuple[float, float] =  (.2, .8) # Bounds for the central bucket when discretizing into -, 0, +
    discretize_clamp: bool = True

    # One-sided heatmaps
    clamp_positive_center: int = 6 # Positive side center
    clamp_positive_radius: int = 5 # Positive side radius
    clamp_positive_minimum: float = .7 # Above this is considered positive; in [0, 1]

    clamp_negative_center: int = 0 # Negative side center
    clamp_negative_radius: int = 5 # Negative side radius
    clamp_negative_maximum: int = .3 # Below this is considered negative; in [0, 1]

class HeatmapVisualizer:
    def __init__(
        self,
        concept_kb: ConceptKB,
        dino_fe: DINOFeatureExtractor,
        config: HeatmapVisualizerConfig = HeatmapVisualizerConfig(),
        rembg_session: rembg.sessions.BaseSession = new_session('isnet-general-use')
    ):
        self.concept_kb = concept_kb
        self.dino_fe = dino_fe
        self.config = config
        self.rembg_session = rembg_session

    def get_difference_heatmap_visualizations(self, concept1: Concept, concept2: Concept, img: Image.Image) -> tuple[Image.Image, Image.Image]:
        img_mask = self._get_foreground_mask(img)

        concept1_heatmap = self._get_heatmap(concept1, img)
        concept2_heatmap = self._get_heatmap(concept2, img)

        concept1_minus_concept2_heatmap = np.clip(concept1_heatmap - concept2_heatmap, 0, 1)
        concept2_minus_concept1_heatmap = np.clip(concept2_heatmap - concept1_heatmap, 0, 1)

        concept1_minus_concept2_vis = Image.fromarray(self._get_heatmap_visualization(img, concept1_minus_concept2_heatmap, img_mask))
        concept2_minus_concept1_vis = Image.fromarray(self._get_heatmap_visualization(img, concept2_minus_concept1_heatmap, img_mask))

        return concept1_minus_concept2_vis, concept2_minus_concept1_vis

    def get_positive_heatmap_visualization(self, concept: Concept, img: Image.Image) -> Image.Image:
        if self.config.strategy != HeatmapStrategy.CLAMP:
            raise NotImplementedError('Only clamp strategy is supported for positive heatmaps')

        img_mask = self._get_foreground_mask(img)
        heatmap = self._get_concept_heatmap(concept, img)
        heatmap = self._get_clamp_heatmap(
            heatmap,
            discretize=False,
            clamp_center=self.config.clamp_positive_center,
            clamp_radius=self.config.clamp_positive_radius
        )

        # Discretize around positive side
        discretized_heatmap = torch.zeros_like(heatmap)
        discretized_heatmap[heatmap > self.config.clamp_positive_minimum] = 1

        heatmap_vis = Image.fromarray(self._get_heatmap_visualization(img, discretized_heatmap, img_mask))

        return heatmap_vis

    def get_negative_heatmap_visualization(self, concept: Concept, img: Image.Image) -> Image.Image:
        if self.config.strategy != HeatmapStrategy.CLAMP:
            raise NotImplementedError('Only clamp strategy is supported for positive heatmaps')

        img_mask = self._get_foreground_mask(img)
        heatmap = self._get_concept_heatmap(concept, img)
        heatmap = self._get_clamp_heatmap(
            heatmap,
            discretize=False,
            clamp_center=self.config.clamp_negative_center,
            clamp_radius=self.config.clamp_negative_radius
        )

        # Discretize around negative side
        discretized_heatmap = torch.zeros_like(heatmap)
        discretized_heatmap[heatmap < self.config.clamp_negative_maximum] = 1

        heatmap_vis = Image.fromarray(self._get_heatmap_visualization(img, discretized_heatmap, img_mask))

        return heatmap_vis

    def get_heatmap_visualization(self, concept: Concept, img: Image.Image) -> Image.Image:
        heatmap = self._get_heatmap(concept, img)
        img_mask = self._get_foreground_mask(img)
        heatmap_vis = Image.fromarray(self._get_heatmap_visualization(img, heatmap, img_mask))

        return heatmap_vis

    def _get_heatmap(self, concept: Concept, img: Image.Image):
        '''
            Returns torch.Tensor of shape (h, w) representing the heatmap for the concept applied to the image.
            This is the raw heatmap score matrix with values in [0, 1], not the visualization.
        '''
        img_mask = self._get_foreground_mask(img)
        heatmap = self._get_concept_heatmap(concept, img)
        heatmap = self._process_heatmap(heatmap, img_mask)

        return heatmap

    def _get_heatmap_visualization(self, img: Image.Image, heatmap: torch.Tensor, img_mask: np.ndarray) -> np.ndarray:
        '''
            Returns np.ndarray of shape (h, w, 3) representing the heatmap visualization.
        '''
        img = np.array(img)

        if self.config.strategy == HeatmapStrategy.NORMALIZE:
            heatmap_vis: np.ndarray = colormaps['rainbow'](heatmap)[..., :3] # (h, w) --> (h, w, 4) --> (h, w, 3)
            heatmap_vis = self.config.opacity * heatmap_vis + (1 - self.config.opacity) * img / 255 # Blend with original image

        elif self.config.strategy == HeatmapStrategy.CLAMP:
            heatmap_vis: np.ndarray = colormaps['viridis'](heatmap)[..., :3] # (h, w) --> (h, w, 4) --> (h, w, 3)
            # heatmap_vis = colormaps['bwr'](heatmap)[..., :3] # (h, w) --> (h, w, 4) --> (h, w, 3)
            heatmap_vis = self.config.opacity * heatmap_vis + (1 - self.config.opacity) * img / 255

        else:
            assert self.config.strategy == HeatmapStrategy.HSV

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

        return (heatmap_vis * 255).astype(np.uint8)

    def _process_heatmap(self, heatmap: torch.Tensor, img_mask: np.ndarray):
        if self.config.strategy == HeatmapStrategy.NORMALIZE:
            heatmap = self._get_normalize_heatmap(heatmap, img_mask)
        elif self.config.strategy == HeatmapStrategy.CLAMP:
            heatmap = self._get_clamp_heatmap(heatmap)
        else:
            assert self.config.strategy == HeatmapStrategy.HSV

        return heatmap

    def _get_concept_heatmap(self, concept: Concept, img: Image.Image):
        feature_predictor = nn.Sequential(
            concept.predictor.img_features_predictor,
            concept.predictor.img_features_weight
        ).eval().cpu()

        # Patch features
        _, patch_feats = get_rescaled_features(self.dino_fe, [img], interpolate_on_cpu=True)
        patch_feats = patch_feats[0] # (resized_h, resized_w, d)
        patch_feats = rescale_features(patch_feats, img) # (h, w, d)

        # Get heatmap
        with torch.no_grad():
            # Need to move to CPU otherwise runs out of GPU mem on big images
            heatmap = feature_predictor(patch_feats.cpu()).squeeze() # (h, w)

        # Move img_features_predictor back to correct device (train is called by train method)
        feature_predictor.cuda()

        return heatmap

    def _get_normalize_heatmap(self, heatmap: torch.Tensor, img_mask: np.ndarray):
        fg_vals = heatmap[img_mask] # Foreground values
        min_val = fg_vals.min()
        max_val = fg_vals.max()

        heatmap = (heatmap - min_val) / (max_val - min_val) # Normalize

        return heatmap

    def _get_clamp_heatmap(
        self,
        heatmap: torch.Tensor,
        clamp_center: float = None,
        clamp_radius: float = None,
        clamp_discretize_lower_bound: float = None,
        clamp_discretize_upper_bound: float = None,
        discretize: bool = None
    ):
        logger.debug(f'Heatmap (min, max): {heatmap.min().item():.2f}, {heatmap.max().item():.2f}')

        if discretize is None:
            discretize = self.config.discretize_clamp
        if clamp_center is None:
            clamp_center = self.config.clamp_center
        if clamp_radius is None:
            clamp_radius = self.config.clamp_radius
        if clamp_discretize_lower_bound is None or clamp_discretize_upper_bound is None:
            clamp_discretize_lower_bound = self.config.clamp_discretize_bounds[0]
            clamp_discretize_upper_bound = self.config.clamp_discretize_bounds[1]

        heatmap = heatmap - clamp_center # Center at zero
        heatmap = heatmap.clamp(-clamp_radius, clamp_radius) # Restrict to [-radius, radius]
        heatmap = (heatmap + clamp_radius) / (2 * clamp_radius) # Normalize to [0, 1] with zero at .5

        if discretize:
            new_heatmap = torch.full_like(heatmap, fill_value=.5)
            new_heatmap[heatmap < clamp_discretize_lower_bound] = 0
            new_heatmap[heatmap > clamp_discretize_upper_bound] = 1

            heatmap = new_heatmap

        return heatmap

    def _get_foreground_mask(self, img: Image.Image, remove_background: bool = None):
        if remove_background is None:
            remove_background = self.config.remove_background

        if remove_background:
            bw_img = remove(img, post_process_mask=True, session=self.rembg_session, only_mask=True) # single-channel image
            img_mask = np.array(bw_img) > 0 # (h, w)

        else:
            img_mask = np.ones((img.height, img.width), dtype=bool)

        return img_mask

# %%
if __name__ == '__main__':
    from feature_extraction import build_feature_extractor, build_sam, build_clip, build_dino
    from image_processing import build_localizer_and_segmenter
    import coloredlogs

    coloredlogs.install(level='DEBUG', logger=logger)

    ckpt_path = '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_06_05-20:23:53-yd491eo3-all_planes_and_guns-infer_localize/concept_kb_epoch_26.pt'

    concept_kb = ConceptKB.load(ckpt_path)

    loc_and_seg = build_localizer_and_segmenter(build_sam(), None) # Save time by not loading DesCo
    clip = build_clip()
    feature_extractor = build_feature_extractor(dino_model=build_dino(), clip_model=clip[0], clip_processor=clip[1])
    dino_fe = feature_extractor.dino_feature_extractor

    # %%
    image_path1 = concept_kb['assault rifle'].examples[2].image_path
    image_path2 = concept_kb['fighter jet'].examples[0].image_path

    config = HeatmapVisualizerConfig(
        strategy=HeatmapStrategy.CLAMP,
        discretize_clamp=True,

        clamp_radius=5,
        clamp_center=6,
        clamp_discretize_bounds=(.2, .8),

        clamp_positive_center=6,
        clamp_positive_radius=5,
        clamp_positive_minimum=.7,

        clamp_negative_center=0,
        clamp_negative_radius=5,
        clamp_negative_maximum=.3,
    )

    visualizer = HeatmapVisualizer(concept_kb, dino_fe, config=config)
    self = visualizer # For debugging

    concept1 = concept_kb['assault rifle']
    concept2 = concept_kb['sniper rifle']
    img = Image.open(image_path1).convert('RGB')

    # visualizer.get_positive_heatmap_visualization(concept1, img)
    # visualizer.get_negative_heatmap_visualization(concept1, img)
    # visualizer.get_heatmap_visualization(concept1, img)
    map1, map2 = visualizer.get_difference_heatmap_visualizations(concept1, concept2, img)
# %%
