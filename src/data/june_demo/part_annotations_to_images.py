import os
import torch
import jsonargparse as argparse
import pandas as pd
import numpy as np
from typing import Literal
from PIL import Image
from tqdm import tqdm
from torchvision.ops import masks_to_boxes, box_convert
from torchvision.transforms.functional import crop, to_pil_image
import logging

logger = logging.getLogger(__file__)

class ImageGenerator:
    def __init__(self, class_root_dir: str, strategy: Literal['crop', 'mask_bg'] = 'crop'):
        self.class_root_dir = class_root_dir
        self.strategy = strategy
        self.class_name = os.path.basename(class_root_dir)
        self.segmentation_class_to_rgb = self._load_segmentation_class_to_rgb(class_root_dir)

    def generate_images(self, class_output_dir: str):
        original_image_dir = os.path.join(self.class_root_dir, 'ImageSets', 'original_images')
        if not os.path.exists(original_image_dir):
            raise FileNotFoundError(f'Image directory not found at: {original_image_dir}')

        # Create segmentation class subdirectories
        for segmentation_class in self.segmentation_class_to_rgb:
            os.makedirs(os.path.join(class_output_dir, segmentation_class), exist_ok=True)

        # Output segmentation class images
        original_image_paths = self._list_image_paths(original_image_dir)
        segmentation_image_paths = [
            os.path.join(
                self.class_root_dir,
                'SegmentationClass',
                self.class_name,
                os.path.splitext(os.path.basename(original_image_path))[0] + '.png'
            )
            for original_image_path in original_image_paths
        ]

        for original_image_path, segmentation_image_path in tqdm(zip(original_image_paths, segmentation_image_paths), total=len(original_image_paths)):
            original_image = Image.open(original_image_path).convert('RGB')
            segmentation_image = Image.open(segmentation_image_path).convert('RGB')

            for segmentation_class in self.segmentation_class_to_rgb:
                try:
                    seg_image = self._get_segmentation_class_image(original_image, segmentation_image, segmentation_class)
                except RuntimeError as e:
                    logger.debug(f'Failed to get segmentation class {segmentation_class} for image at {segmentation_image_path}')
                    continue

                output_path = os.path.join(class_output_dir, segmentation_class, os.path.basename(original_image_path))
                seg_image.save(output_path)

    def _get_segmentation_class_image(self, original_image: Image.Image, segmentation_image: Image.Image, segmentation_class: str) -> Image.Image:
        image_array = np.array(original_image)

        segmentation_image_array = np.array(segmentation_image)
        class_mask = segmentation_image_array == self.segmentation_class_to_rgb[segmentation_class] # (h, w, 3)
        class_mask = np.prod(class_mask, axis=-1).astype(bool) # (h, w)

        if class_mask.sum() == 0:
            raise RuntimeError(f'Class {segmentation_class} not found in segmentation image')

        if self.strategy == 'crop':
            return self._get_cropped_image(image_array, class_mask)
        if self.strategy == 'mask_bg':
            return self._get_masked_image(image_array, class_mask)
        else:
            raise NotImplementedError(f'Strategy not implemented: {self.strategy}')

    def _get_cropped_image(self, image_array: np.ndarray, mask: np.ndarray) -> Image.Image:
        boxes = masks_to_boxes(torch.from_numpy(mask)[None,...]) # (1,4): (x1, y1, x2, y2)
        x, y, w, h = box_convert(boxes, 'xyxy', 'xywh')[0].int().tolist()
        cropped_image = crop(torch.from_numpy(image_array).permute(2, 0, 1), y, x, h, w)

        return to_pil_image(cropped_image)

    def _get_masked_image(self, image_array: np.ndarray, mask: np.ndarray) -> Image.Image:
        bg_mask = np.logical_not(mask)
        bg_mask = np.broadcast_to(bg_mask[..., None], (bg_mask.shape[0], bg_mask.shape[1], 3))
        image_array[bg_mask] = 255

        return Image.fromarray(image_array)

    def _load_segmentation_class_to_rgb(self, class_root_dir: str) -> dict[str, np.ndarray]:
        class_to_rgb_file_path = os.path.join(class_root_dir, 'labelmap.txt')

        if not os.path.exists(class_to_rgb_file_path):
            raise FileNotFoundError(f'File not found: {class_to_rgb_file_path}')

        df = pd.read_csv(class_to_rgb_file_path, sep=':', header=0, names=['label', 'rgb', 'parts', 'actions'])
        class_to_rgb = {
            row.label : np.array(list(map(int, row.rgb.split(','))))
            for row in df.itertuples()
        }

        return class_to_rgb

    def _list_image_paths(self, image_dir: str, exts=['.jpg', '.png']):
        paths = []
        for dirpath, dirnames, filenames in os.walk(image_dir):
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() in exts:
                    paths.append(os.path.join(dirpath, filename))

        return paths

def parse_args(cl_args: list[str] = None, config_str: str = None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--strategy', choices=['crop', 'mask_bg'], default='crop')
    parser.add_argument('--image_classes_root',
                        default='/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplane_part_annotations/biplane/SegmentationClass/biplane/000001.png')
    parser.add_argument('--output_dir', required=True)

    if config_str:
        args = parser.parse_string(config_str)
    else:
        args = parser.parse_args(cl_args)

    return args, parser

if __name__ == '__main__':
    import coloredlogs
    coloredlogs.install(level='DEBUG')

    args, parser = parse_args(config_str='''
        strategy: mask_bg
        image_classes_root: /shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplane_part_annotations
        output_dir: /shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplane_part_masked_bg
    ''')

    for base in os.listdir(args.image_classes_root):
        path = os.path.join(args.image_classes_root, base)
        if not os.path.isdir(path):
            continue

        class_root_dir = path
        class_name = base

        logger.info(f'Generating images for class: {class_name}')

        image_generator = ImageGenerator(class_root_dir, args.strategy)

        class_output_dir = os.path.join(args.output_dir, class_name)
        image_generator.generate_images(class_output_dir)