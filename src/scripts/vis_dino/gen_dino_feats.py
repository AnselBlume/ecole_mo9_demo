# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys
sys.path.append('/shared/nas2/blume5/fa23/ecole/src/mo9_demo/src')

import torch
import torch.nn.functional as F
import os
from feature_extraction import build_dino, DinoFeatureExtractor
import jsonargparse as argparse
import pickle
from PIL import Image
from einops import rearrange

def parse_args(cl_args = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('img_paths', nargs='+', help='Paths to images')
    parser.add_argument('--output_basename', default='features.pkl')
    parser.add_argument('--output_dir', default='/shared/nas2/blume5/fa23/ecole')

    return parser.parse_args(cl_args)

def rescale_features(features: torch.Tensor, img: Image.Image = None, width: int = None, height: int = None):
    '''
        features: (n, h_patch, w_patch, d) or (h_patch, w_patch, d)

        Returns: Interpolated features to the size of the image.
    '''
    assert bool(img) ^ bool(width and height), 'Exactly one of img or (width and height) must be provided'

    if img:
        width, height = img.size

    if len(features.shape) == 3:
        features = features.unsqueeze(0)

    features = F.interpolate(
        rearrange(features, 'n h w d -> n d h w'),
        size=(height, width),
        mode='bilinear'
    )

    features = rearrange(features, 'n d h w -> n h w d')

    return features

if __name__ == '__main__':
    args = parse_args([
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/xiaomeng_augmented_data_v3/bowl_1.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/xiaomeng_augmented_data_v3/bowl_14.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/xiaomeng_augmented_data_v3/bowl_18.jpg'
    ])

    # %%
    resize_images = True
    feature_extractor = DinoFeatureExtractor(build_dino(), resize_images=resize_images)

    # %%
    imgs = [Image.open(p) for p in args.img_paths]

    with torch.no_grad():
        cls_feats, patch_feats = feature_extractor(imgs) # (n_imgs, n_features)

    cls_feats = cls_feats.cpu()

    # Rescale patch features
    if resize_images: # All images are the same size
        patch_feats = rescale_features(patch_feats, imgs[0]).cpu()

    else:
        patch_feats = [
            rescale_features(patch_feat, img).cpu()
            for patch_feat, img in zip(patch_feats, imgs)
        ]

    # %%
    out_dict = {
        'cls_features': cls_feats,
        'patch_features': patch_feats
    }

    out_path = os.path.join(args.output_dir, args.output_basename)
    with open(out_path, 'wb') as f:
        pickle.dump(out_dict, f)

    print('Features dumped to:', out_path)