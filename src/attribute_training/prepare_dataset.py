"""
Segment and crop VAW dataset. Save output to new folder
"""
import sys
import os 
from tqdm import tqdm 
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
def open_file(filename,json=True):
        if json:
            with open(filename) as fopen:
                contents = json.load(fopen)
        else:
            with open(filename,'rb') as fopen:
                contents = pickle.load(fopen)
        return contents 
def convert_annotations(old_annotation_name,new_annotation_name):
    """
    Convert annotations from list to dictionary where key is instance id
    """
    new_annotations = {}
    with open(old_annotation_name,'r+') as fopen:
        old_annotations = json.load(fopen)
    for entry in tqdm(old_annotations):
        instance_id = entry['instance_id']
        new_annotations[instance_id] = entry 
    with open(new_annotation_name,'w+') as fwrite:
        json.dump(fwrite,new_annotation_name)
def count_samples_per_annotation(annotation_file,attribute_to_id):
    samples = np.zeros(620)
    annotations = json.load(annotation_file)

    with open(annotation_file,'r+') as fopen:
        annotations = json.load(fopen)
    
    for entry in tqdm(annotations):
        positive_attributes = entry['positive_attributes']

convert_annotations('/scratch/bcgp/datasets/visual_genome/vaw_dataset/data/all_train_old.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/data/train.json')
convert_annotations('/scratch/bcgp/datasets/visual_genome/vaw_dataset/data/old_val.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/data/val.json')
convert_annotations('/scratch/bcgp/datasets/visual_genome/vaw_dataset/data/old_test.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/data/test.json')

#prepare_images('/scratch/bcgp/datasets/visual_genome/vaw_dataset/data/val.json','/scratch/bcgp/datasets/visual_genome/images','/scratch/bcgp/datasets/vaw_cropped/val')


    
