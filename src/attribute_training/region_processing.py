import torch
import os
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils
import torch
from PIL import Image 
import torchvision.transforms as T
import math
import os
import argparse
import torch.nn.functional as F
import cv2 
import extract_features as image_features
import logging
import json 
import pickle 
from json import JSONEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

"""
Given extracted regions from SAM, extract image features (if not done already), create feature vectors for each region using some method (eg. avg)
"""

def save_file(filename,data,json_numpy=False):
    """
    Based on https://github.com/salesforce/LAVIS/blob/main/lavis/common/utils.py
    Supported:
        .pkl, .pickle, .npy, .json
    """
    parent_dir = os.path.dirname(filename)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    # Path(parent_dir).chmod(0o0777)
    file_ext = os.path.splitext(filename)[1]
    if file_ext == ".npy":
        with open(filename, "wb+") as fopen:
            np.save(fopen, data)
    elif file_ext == ".json":
        with open(filename,'w+') as fopen:
                json.dump(data,fopen,indent=2)        
    else:
        # assume file is pickle
         with open(filename, "wb+") as fopen:
            pickle.dump(data, fopen)




def open_file(filename):
    """
    Based on https://github.com/salesforce/LAVIS/blob/main/lavis/common/utils.py
    Supported:
        .pkl, .pickle, .npy, .json
    """
    file_ext = os.path.splitext(filename)[1]
    if file_ext == '.txt':
        with open(filename,'r+') as fopen:
            data = fopen.readlines()
    elif file_ext in [".npy",".npz"]:
        data = np.load(filename,allow_pickle=True)
    elif file_ext == '.json':
        with open(filename,'r+') as fopen:
            data = json.load(fopen)
    else:
        # assume pickle
        with open(filename,"rb+") as fopen:
            data = pickle.load(fopen)
    return data

def region_features(args,image_id_to_sam):
    if args.feature_dir!= None:
        features_exist = True 
        # Get the intersection of the feature files and the sam regions
        all_feature_files = [f for f in os.listdir(args.feature_dir) if os.path.isfile(os.path.join(args.feature_dir, f))]
        feature_files_in_sam = [f for f in all_feature_files if os.path.splitext(f)[0] in image_id_to_sam]

        features_minus_sam = set(all_feature_files) - set(feature_files_in_sam)
        # if len(features_minus_sam) > 0:
        #     logger.warning(f'Found {len(features_minus_sam)} feature files that are not in the set of SAM region files: {features_minus_sam}')
    else:
        features_exist = False 
        logger.warning('No feature directory. Will extract features while processing features')
    if features_exist:
        prog_bar = tqdm(feature_files_in_sam)

    else:
        prog_bar = tqdm(image_id_to_sam)
    bad_mask = []
    def extract_features(f, args,device='cuda',features_exist=True):
        prog_bar.set_description(f'Region features: {f}')

        features = open_file(os.path.join(args.feature_dir,f))

        file_name = f
        ext = os.path.splitext(f)[1]
        all_region_features_in_image = []
        sam_regions = image_id_to_sam[file_name.replace(ext,'')]
        
        if args.pooling_method == 'downsample':
            f1, h1, w1 = features[0].shape

            for region in sam_regions:
                sam_region_feature = {}
                sam_region_feature['region_id'] = region['region_id']
                sam_region_feature['area'] = region['area']
                sam_mask = mask_utils.decode(region['segmentation'])
                h2, w2 = sam_mask.shape
                downsampled_mask = torch.from_numpy(sam_mask).cuda()
                downsampled_mask = downsampled_mask.unsqueeze(0).unsqueeze(0)
                downsampled_mask = torch.nn.functional.interpolate(downsampled_mask, size=(h1, w1), mode='nearest').squeeze(0).squeeze(0)

                if torch.sum(downsampled_mask).item() == 0:
                    continue

                features_in_sam = torch.from_numpy(features).cuda().squeeze(dim = 0)[:, downsampled_mask==1].view(f1, -1).mean(1).cpu().numpy()
                sam_region_feature['region_feature'] = features_in_sam
                all_region_features_in_image.append(sam_region_feature)
        else:
            if len(sam_regions) > 0:
                # sam regions within an image all have the same total size
                new_h, new_w = mask_utils.decode(sam_regions[0]['segmentation']).shape
           
                patch_length = args.dino_patch_length
                padded_h, padded_w = math.ceil(new_h / patch_length) * patch_length, math.ceil(new_w / patch_length) * patch_length # Get the padded height and width
                upsample_feature = torch.nn.functional.interpolate(torch.from_numpy(features).cuda(), size=[padded_h,padded_w],mode='bilinear') # First interpolate to the padded size
                upsample_feature = T.CenterCrop((new_h, new_w)) (upsample_feature).squeeze(dim = 0) # Apply center cropping to the original size
                f,h,w = upsample_feature.size()
               

                for region in sam_regions:
                    sam_region_feature = {}
                    sam_region_feature['instance_id'] = region['instance_id']
                    sam_mask = mask_utils.decode(region['segmentation'])
                    sam_h, sam_w = sam_mask.shape 
                    # if sam_h != h or sam_w!= w:
                    #     bad_mask.append((f,region['instance_id']))
                    #     patch_length = args.dino_patch_length
                    #     padded_h, padded_w = math.ceil(new_h / patch_length) * patch_length, math.ceil(new_w / patch_length) * patch_length # Get the padded height and width
                    #     upsample_feature = torch.nn.functional.interpolate(torch.from_numpy(features).cuda(), size=[padded_h,padded_w],mode='bilinear') # First interpolate to the padded size
                    #     upsample_feature = T.CenterCrop((new_h, new_w)) (upsample_feature).squeeze(dim = 0) # Apply center cropping to the original size
                    #     f,h,w = upsample_feature.size()
               


                    r_1, r_2 = np.where(sam_mask.astype(np.int32) == 1)

                    if args.pooling_method == 'average':
                        try:
                            features_in_sam = upsample_feature[:,r_1,r_2].view(f,-1).mean(1).cpu().numpy()
                        except:
                            print(f'Failed for {file_name}')
                    elif args.pooling_method == 'max':
                        input_max, max_indices = torch.max(upsample_feature[:,r_1,r_2].view(f,-1), 1)
                        features_in_sam = input_max.cpu().numpy()

                    sam_region_feature['region_feature'] = features_in_sam
                    save_file(os.path.join(args.region_feature_dir,region['instance_id']+'.pkl'),sam_region_feature)
                    #all_region_features_in_image.append(sam_region_feature)
        #save_file(os.path.join(args.region_feature_dir, file_name.replace(ext,'.pkl')), all_region_features_in_image)

    for i,f in enumerate(prog_bar):
        try:
            extract_features(f,args,features_exist=features_exist)

        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f'Caught CUDA out of memory error for {f}; falling back to CPU')
            torch.cuda.empty_cache()
            extract_features(f,args,features_exist=features_exist, device='cpu')
        # except Exception as e:
        #     print(f'Error: {e}')
        #     continue 


def load_all_regions(args):
    if len(os.listdir(args.mask_dir)) == 0:
        raise Exception(f"No regions found at {args.mask_dir}")
    logger.info(f"Loading region masks from {args.mask_dir}")
    image_id_to_mask = {}
    for f in tqdm(os.listdir(args.mask_dir)):
        filename_extension = os.path.splitext(f)[1]
        regions = open_file(os.path.join(args.mask_dir,f))
        if not args.use_sam:
            regions = [r for r in regions if 'mask' in list(r.keys())]
        image_id_to_mask[f.replace(filename_extension,'')] = regions
    return image_id_to_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main_dir",
        type=str,
        default="/shared/rsaas/dino_sam"
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=None,
        help="Location of extracted features",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="Location of masks (sam or ground truth if given)",
    )

    parser.add_argument(
        "--region_feature_dir",
        type=str,
        default=None,
        help="Location of features per region/pooled features",
    )

    parser.add_argument(
        "--dino_patch_length",
        type=int,
        default=14,
        help="the length of dino patch",
    )

    parser.add_argument(
        "--use_sam",
        action="store_false",
        help="If not using json sam regions"
    )
    
    parser.add_argument(
        "--pooling_method",
        type=str,
        default='average',
        choices=['average', 'max', 'downsample'],
        help='pooling methods'
    )

    # extract feature arguments 
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help='Image dir for extracting features'
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='bf16',
        choices=['fp16', 'fp32','bf16'],
        help="Which mixed precision to use. Use fp32 for clip and dense_clip"
    )
    
    args = parser.parse_args()


    image_id_to_mask = load_all_regions(args)
    region_features(args,image_id_to_mask)

    logger.info('Done')