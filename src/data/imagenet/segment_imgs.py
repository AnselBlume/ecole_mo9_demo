'''
    Generates SAM masks for the images in a directory and saves them in a directory with the same structure.
    Intended to be run using the torchsam environment (or any environment with segment_anything_fast installed)
'''
import argparse
from segment_anything_fast import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
from einops import rearrange
import json
from tqdm import tqdm
import os
from pycocotools import mask as mask_utils
from PIL import Image
import torch

def parse_args(cl_args = None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dir', help='Root directory of the images from which the DesCo boxes were detected.')
    parser.add_argument('--out_dir', help='Output directory for the rle JSONs to be built in the same structure as --boxes_dir')
    parser.add_argument('--device', help='Torch device to use')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for multiprocessing to shuffle paths.')

    return parser.parse_args(cl_args)

def get_paths(root_dir: str, use_dirs=False, must_contain='', has_ext=''):
    paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if use_dirs:
            filenames = dirnames

        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if must_contain in file_path and filename.endswith(has_ext):
                paths.append(file_path)

                return paths

    return sorted(paths)

def map_path_to_other_dir(path: str, curr_dir: str, other_dir: str, new_ext=''):
    '''
        Maps a path in curr_dir to the same relative path in other_dir.

        path (str): Path to the file which is a descendent of curr_dir to map to other_dir.
        curr_dir (str): Ancestor directory (not necessarily direct parent) of path. Used to extract relative path of path wrt curr_dir.
        other_dir (str): Directory to map relative path of path in curr_dir to.
        new_ext: The new extension to replace the old one with, including the dog (e.g. ".json").
    '''
    rel_path = path.split(curr_dir)[-1].strip('/')
    path = os.path.join(other_dir, rel_path)

    path_no_ext, old_ext = os.path.splitext(path)
    path = f'{path_no_ext}{new_ext if new_ext else old_ext}'

    return path

def dump_masks(masks: np.ndarray, output_path):
    if masks.shape[0] == 0:
        rles = []

    else:
        rles = mask_utils.encode(np.asfortranarray(rearrange(masks, 'b h w -> h w b')))
        for rle in rles:
            rle['counts'] = rle['counts'].decode() # bin -> str

    with open(output_path, 'w') as f:
        json.dump(rles, f)

def apply_eval_dtype_sam(model, dtype):
    '''
        Copied from https://github.com/pytorch-labs/segment-anything-fast/blob/387488bc4c7ab2ae311fb0632b34cab5cbfbab78/segment_anything_fast/build_sam.py#L54
    '''

    def prep_model(model, dtype):
        if dtype is not None:
            return model.eval().to(dtype)
        return model.eval()

    model.image_encoder = prep_model(model.image_encoder, dtype)
    model.prompt_encoder = prep_model(model.prompt_encoder, dtype)
    model.mask_decoder = prep_model(model.mask_decoder, dtype)

    return model

if __name__ == '__main__':
    args = parse_args([
        '--img_dir', '/shared/nas2/blume5/fa23/ecole/data/imagenet/subset-whole_unit-100',
        '--out_dir', '/shared/nas2/blume5/fa23/ecole/data/imagenet/subset-whole_unit-100-sam_masks',
        '--device', 'cuda:1',
        '--random_seed', '0',
    ])

    # %%

    # %% # Generate segmentations for boxes_dir
    sam = sam_model_registry['vit_h'](checkpoint='/shared/nas2/blume5/fa23/ecole/checkpoints/sam/sam_vit_h_4b8939.pth')
    sam.eval()
    sam.to(args.device)
    # apply_eval_dtype_sam(sam, torch.bfloat16)

    sam_amg: SamAutomaticMaskGenerator = SamAutomaticMaskGenerator( # Part-based SAM
        model=sam,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
        process_batch_size=8
    )

    rng = np.random.default_rng(args.random_seed)

    print('Collecting paths...')
    paths = get_paths(args.img_dir)
    rng.shuffle(paths)

    print('Generating segmentations')
    prog_bar = tqdm(paths)
    for path in prog_bar:
        rel_path = map_path_to_other_dir(path, args.img_dir, '')
        prog_bar.set_description(rel_path)

        img = np.array(Image.open(path).convert('RGB'))

        with torch.inference_mode():
            masks = sam_amg.generate(img)

        out_path = map_path_to_other_dir(path, args.img_dir, args.out_dir, '.json')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        dump_masks(masks, out_path)