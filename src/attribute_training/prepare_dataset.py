"""
Segment and crop VAW dataset. Save output to new folder
"""
import sys
import os 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import os 
import json 
import cv2 
from tqdm import tqdm 
from typing import Optional
import vis_utils
from vis_utils import mask_and_crop_image

def prepare_images(json_file:str,image_dir:str,save_dir:str):
    with open(json_file,'r+') as fopen:
        annotation_list = json.load(fopen)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i,entry in enumerate(tqdm(annotation_list)):
        image_id = entry['image_id']
        instance_id = entry['instance_id']
        bbox = entry['instance_bbox']
        mask = entry['instance_polygon']
        if mask!= None:
            try:
                segmented_and_cropped_img = mask_and_crop_image(os.path.join(image_dir,image_id)+'.jpg',mask[0])
                cv2.imwrite(os.path.join(save_dir,instance_id)+'.jpg',segmented_and_cropped_img)
            except:
                print('wrong')
                continue 




prepare_images('/scratch/bcgp/datasets/visual_genome/vaw_dataset/data/all_train.json','/scratch/bcgp/datasets/visual_genome/images','/scratch/bcgp/datasets/vaw_cropped')


    
