import os

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys

sys.path = ['/shared/nas2/blume5/fa23/ecole/src/mo9_demo/src'] + sys.path

import logging
import pickle

import coloredlogs
from feature_extraction import build_sam
from image_processing import build_localizer_and_segmenter
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__file__)
coloredlogs.install(level=logging.INFO, logger=logger)

if __name__ == '__main__':
    in_dirs = [
        '/shared/nas2/blume5/fa23/ecole/data/imagenet/negatives_rand_1k_cropped',
        '/shared/nas2/blume5/fa23/ecole/data/imagenet/negatives_rand_1k'
    ]
    out_dirs = [
        '/shared/nas2/blume5/fa23/ecole/cache/imagenet_rand_1k_cropped',
        '/shared/nas2/blume5/fa23/ecole/cache/imagenet_rand_1k'
    ]

    loc_and_seg = build_localizer_and_segmenter(build_sam(), None)

    for in_dir, out_dir in zip(in_dirs, out_dirs):
        logger.info(f'Processing {in_dir} -> {out_dir}')
        prog_bar = tqdm(range(1000))

        for dirpath, dirnames, filenames in os.walk(in_dir):
            for filename in filenames:
                if os.path.splitext(filename)[1] not in ['.jpg', '.png']:
                    continue

                prog_bar.update(1)

                in_path = os.path.join(dirpath, filename)
                out_path = os.path.join(out_dir, os.path.relpath(in_path, in_dir))
                out_path = os.path.splitext(out_path)[0] + '.pkl'

                if os.path.exists(out_path):
                    continue

                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                try:
                    image = Image.open(in_path).convert('RGB')
                    segmentations = loc_and_seg.localize_and_segment(image)
                    segmentations.input_image_path = in_path

                    with open(out_path, 'wb') as f:
                        pickle.dump(segmentations, f)

                except Exception as e:
                    logger.error(f'Error processing {in_path}: {e}')
                    continue
