import os 
import torch 
import pickle
import os 
import sys
import numpy as np
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

def extract_clip(args, model, image, preprocess=None):
     image_input = preprocess(image).unsqueeze(0).to(device=args.device)
     with torch.no_grad():
        features = model.encode_image(image_input)
     return features.detach().cpu().to(torch.float32).numpy()

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
        features = extract_clip(args, model, image, preprocess)
        if not os.path.exists(args.feature_dir):
            os.makedirs(args.feature_dir)
        with open(os.path.join(args.feature_dir,image_name.replace(filename_extension,'.pkl')),'wb+') as fopen:
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
        "--clip_model",
        type=str,
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-L/14@336px"],
        help="CLIP base model version"
    )
    args = parser.parse_args([
        '--image_dir','/scratch/bcgp/datasets/vaw_cropped/val',
        '--feature_dir','/scratch/bcgp/datasets/vaw_cropped/features/val',
        '--clip_model','ViT-L/14'
    ])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device 
    model, preprocess = clip.load(args.clip_model, device=device)
    extract_features(model, args, preprocess)
