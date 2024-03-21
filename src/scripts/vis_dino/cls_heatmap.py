# %%
import os
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
sys.path.append('/shared/nas2/blume5/fa23/ecole/src/mo9_demo/src')

from PIL import Image
from gen_dino_feats import get_rescaled_features
from feature_extraction import build_dino, DINOFeatureExtractor
import matplotlib.pyplot as plt
import torch
from rembg import remove, new_session

def normalize(x: torch.Tensor):
    x = x.squeeze()
    return (x - x.min()) / (x.max() - x.min())

def get_heatmaps(
    feature_extractor: DINOFeatureExtractor,
    image1: Image.Image,
    image2: Image.Image,
    resize_images: bool = True
):

    cls_feats1, patch_feats1 = get_rescaled_features(feature_extractor, [image1], resize_image=resize_images) # (1, d), (1, n_patches_h, n_patches_w, d)
    cls_feats2, patch_feats2 = get_rescaled_features(feature_extractor, [image2], resize_image=resize_images)

    if isinstance(patch_feats1, list): # If the image is not cropped
        patch_feats1 = patch_feats1[0].unsqueeze(0)
        patch_feats2 = patch_feats2[0].unsqueeze(0)

    cls_one_minus_two = cls_feats1 - cls_feats2 # (1, d)
    cls_two_minus_one = cls_feats2 - cls_feats1 # (1, d)

    one_minus_two_times_one = normalize(patch_feats1 @ cls_one_minus_two.T) # (1, n_patches_h, n_patches_w)
    one_minus_two_times_two = normalize(patch_feats2 @ cls_one_minus_two.T) # (1, n_patches_h, n_patches_w)

    two_minus_one_times_one = normalize(patch_feats1 @ cls_two_minus_one.T) # (1, n_patches_h, n_patches_w)
    two_minus_one_times_two = normalize(patch_feats2 @ cls_two_minus_one.T) # (1, n_patches_h, n_patches_w)

    return {
        'one_minus_two_times_one': one_minus_two_times_one,
        'one_minus_two_times_two': one_minus_two_times_two,
        'two_minus_one_times_one': two_minus_one_times_one,
        'two_minus_one_times_two': two_minus_one_times_two
    }

def vis_heatmaps(
    heatmaps: dict,
    image1: Image.Image,
    image2: Image.Image,
    figsize=(15, 10),
    cmap='rainbow'
):
    fig, axs = plt.subplots(2, 3, figsize=figsize)

    axs[0, 0].imshow(image1)
    axs[0, 0].set_title('Image 1')
    axs[0, 0].axis('off')

    axs[1, 0].imshow(image2)
    axs[1, 0].set_title('Image 2')
    axs[1, 0].axis('off')

    axs[0, 1].imshow(heatmaps['one_minus_two_times_one'], cmap='rainbow')
    axs[0, 1].set_title('(One - Two) @ One')
    axs[0, 1].axis('off')

    axs[1, 1].imshow(heatmaps['one_minus_two_times_two'], cmap='rainbow')
    axs[1, 1].set_title('(One - Two) @ Two')
    axs[1, 1].axis('off')

    axs[0, 2].imshow(heatmaps['two_minus_one_times_one'], cmap='rainbow')
    axs[0, 2].set_title('(Two - One) @ One')
    axs[0, 2].axis('off')

    axs[1, 2].imshow(heatmaps['two_minus_one_times_two'], cmap='rainbow')
    axs[1, 2].set_title('(Two - One) @ Two')
    axs[1, 2].axis('off')

    fig.suptitle('Heatmaps of DINO features')

    return fig, axs

if __name__ == '__main__':
    # img1_path = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/xiaomeng_augmented_data_v3/fork_1_ori.jpg'
    # img2_path = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/xiaomeng_augmented_data_v3/spoon_3.jpg'

    # img1_path = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/xiaomeng_augmented_data_v3/mug_12.jpg'
    # img2_path = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/xiaomeng_augmented_data_v3/bowl_6.jpg'

    img1_path = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/xiaomeng_augmented_data_v3/hoe_1_original.jpg'
    img2_path = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/xiaomeng_augmented_data_v3/shovel_15.jpg'

    resize_images = False
    remove_bg = True

    # %%
    feature_extractor = DINOFeatureExtractor(build_dino(), resize_images=resize_images)

    # %%
    image1 = Image.open(img1_path).convert('RGB')
    image2 = Image.open(img2_path).convert('RGB')

    if remove_bg:
        rembg_session = new_session('isnet-general-use')
        image1 = remove(image1, session=rembg_session, post_process_mask=True).convert('RGB')
        image2 = remove(image2, session=rembg_session, post_process_mask=True).convert('RGB')

    heatmaps = get_heatmaps(feature_extractor, image1, image2, resize_images=resize_images)
    fig, axs = vis_heatmaps(heatmaps, image1, image2)
# %%
