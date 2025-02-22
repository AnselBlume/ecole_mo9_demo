import os
import torch
import pickle
import os
import sys
import numpy as np
import gzip
import json
import cv2
from tqdm import tqdm
from pycocotools import mask as mask_utils
from PIL import Image
import torchvision.transforms as T
import itertools
import math
import argparse
import torch.nn.functional as F
from pathlib import Path
import clip
import pickle
import shutil
import logging
import subprocess
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
class CenterPadding(torch.nn.Module):
    def __init__(self, multiple = 14):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output

def extract_clip(args, model, image, preprocess=None):
     image_input = preprocess(image).unsqueeze(0).to(device=args.device)
     with torch.no_grad():
        features = model.encode_image(image_input)
     return features.detach().cpu().to(torch.float32).numpy()
def extract_dino_v2(args,model,image):
    layers = eval(args.layers)

    # print(f"Using layers:{layers}")
    if args.padding != "center":
        raise Exception("Only padding center is implemented")
    transform = T.Compose([
        T.ToTensor(),
        lambda x: x.unsqueeze(0),

        CenterPadding(multiple = args.multiple),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    with torch.inference_mode():
        layers = eval(args.layers)
        # intermediate layers does not use a norm or go through the very last layer of output
        img = transform(image).to(device='cuda',dtype=args.dtype)
        features_out = model.get_intermediate_layers(img, n=layers,reshape=True)
        features = torch.cat(features_out, dim=1) # B, C, H, W
    return features.detach().cpu().to(torch.float32).numpy()
def e(args):
    '''
    Save file as tar file
    '''
    dir_name = os.path.dirname(args.feature_dir)
    splits = os.path.split(args.feature_dir)
    command = f'cd {dir_name}; tar -czf {args.feature_dir_save}.tar.gz {splits[-1]}'
    ret = subprocess.run(command,capture_output=True,shell=True)
    logger.info(f'Saved files to {ret}')
def extract_features(model, args, preprocess=None):
    all_image_files = [f for f in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, f))]
    model = model.to(device='cuda')
    model.eval()
    for i, f in enumerate(tqdm(all_image_files, desc='Extract', total=len(all_image_files))):
        image_name = f
        filename_extension = os.path.splitext(image_name)[1]
        cv = cv2.imread(os.path.join(args.image_dir, f))
        try:
            cv = cv2.imread(os.path.join(args.image_dir, f))
            color_coverted = cv2.cvtColor(cv, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(color_coverted)

        except:
            print(f'Could not read image {f}')
            continue
        if args.model == 'clip':
            features = extract_clip(args, model, image, preprocess)
        else:
            features = extract_dino_v2(args,model,image)
        if not os.path.exists(args.feature_dir):
            os.makedirs(args.feature_dir)
        #np.savez_compressed(os.path.join(args.feature_dir,image_name.replace(filename_extension,'.npz')),features=features)
        with open(os.path.join(args.feature_dir,image_name.replace(filename_extension,'.npy')),'wb+') as fopen:
            pickle.dump(features,fopen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Location of jpg files",
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=None,
        help="Location of feature files",
    )
    parser.add_argument(
        "--feature_dir_save",
        type=str,
        default=None,
        help="Location of feature files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='dinov2_vitl14',
        choices=['dinov2_vitl14', 'dino_vitb8', 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitg14', 'clip','dino_vitb16','dense_clip', 'imagenet'],
        help="Name of model from repo"
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-L/14@336px"],
        help="CLIP base model version"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="[23]",
        help="List of layers or number of last layers to take"
    )
    parser.add_argument(
        "--padding",
        default="center",
        help="Padding used for transforms"
    )
    parser.add_argument(
        "--multiple",
        type=int,
        default=14,
        help="The patch length of the model. Use 14 for DINOv2, 8 for DINOv1, 32 for CLIP, 14 for DenseCLIP (automatically handled in the package)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='fp16',
        choices=['fp16','fp32','bf16']
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1000,
        help="Save as tar file every",
    )
    args = parser.parse_args([
        '--image_dir','/scratch/bcgp/datasets/visual_genome/images',
        '--feature_dir','/tmp/features',
        '--dtype','bf16'
    ])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    if args.dtype == "fp16":
      args.dtype = torch.half
    elif args.dtype == "fp32":
      args.dtype = torch.float ## this change is needed for CLIP model
    else:
      args.dtype = torch.bfloat16
    if args.model == 'clip':
        model, preprocess = clip.load(args.clip_model, device=device)
        model = model.to(device=args.device,dtype=args.dtype)
        extract_features(model, args, preprocess)
    else:
        model = torch.hub.load('facebookresearch/dinov2','dinov2_vitb14')
        model = model.to(device=args.device,dtype=args.dtype)
        extract_features(model, args)
