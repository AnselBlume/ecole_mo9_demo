# %%
'''
    Utility functions to generate DINO features for a given image, and to get the rescaled patch
    features at the full image resolution.
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys
sys.path.append('/shared/nas2/blume5/fa23/ecole/src/mo9_demo/src')

import os
import pickle
from feature_extraction import build_dino, DINOFeatureExtractor
import jsonargparse as argparse
from PIL import Image
from feature_extraction.dino_features import get_rescaled_features

def parse_args(cl_args = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('img_paths', nargs='+', help='Paths to images')
    parser.add_argument('--output_basename', default='features.pkl')
    parser.add_argument('--output_dir', default='/shared/nas2/blume5/fa23/ecole')

    return parser.parse_args(cl_args)

if __name__ == '__main__':
    args = parse_args([
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/xiaomeng_augmented_data_v3/bowl_1.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/xiaomeng_augmented_data_v3/bowl_14.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/xiaomeng_augmented_data_v3/bowl_18.jpg'
    ])

    # %%
    resize_images = True
    feature_extractor = DINOFeatureExtractor(build_dino(), resize_images=resize_images)

    # %%
    imgs = [Image.open(p) for p in args.img_paths]
    cls_feats, patch_feats = get_rescaled_features(feature_extractor, [imgs[0]], resize_image=resize_images)

    # %%
    out_dict = {
        'cls_features': cls_feats,
        'patch_features': patch_feats
    }

    out_path = os.path.join(args.output_dir, args.output_basename)
    with open(out_path, 'wb') as f:
        pickle.dump(out_dict, f)

    print('Features dumped to:', out_path)