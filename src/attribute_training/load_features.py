import numpy as np
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
import visualization.vis_utils as vis_utils
from visualization.vis_utils import mask_and_crop_image
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
import skimage
import pickle
def open_file(filename,use_json=True):
        if use_json:
            with open(filename) as fopen:
                contents = json.load(fopen)
        else:
            with open(filename,'rb') as fopen:
                contents = pickle.load(fopen)
        return contents
def save_file(filename,contents):
    with open(filename,'w+') as fwrite:
        json.dump(contents,fwrite)
def load_features(annotation_file:str,attribute_to_index:str,feature_dir:str,save_path:str,file_name:str):
    all_features = []
    all_labels = []
    annotations = open_file(annotation_file)
    attribute_to_id = open_file(attribute_to_index)
    total_labels = len(list(attribute_to_id.keys()))
    for entry in tqdm(list(annotations.values())):
        instance_id = entry['instance_id']
        # label = entry['id']
        labels = np.zeros((total_labels))
        for p in entry['positive_attributes']:
            if p in list(attribute_to_id.keys()):
                attribute_id = attribute_to_id[p]
                labels[attribute_id] = 1
        for n in entry['negative_attributes']:
            if n in list(attribute_to_id.keys()):
                attribute_id = attribute_to_id[n]
                labels[attribute_id] = -1
        if os.path.exists(os.path.join(feature_dir,instance_id+'.pkl')):
            feature_file = open_file(os.path.join(feature_dir,instance_id+'.pkl'),use_json=False)
            if not np.isnan(feature_file['region_feature']).any():

                all_features.append(feature_file['region_feature'])
                all_labels.append(labels)
    all_features = np.stack(all_features)
    all_labels = np.stack(all_labels)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with open(os.path.join(save_path,file_name+'.pkl'),'wb') as fsave:
        pickle.dump({'features':all_features,'labels':all_labels},fsave)
load_features('/scratch/bcgp/datasets/visual_genome/vaw_dataset/data/train.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/color_data_single_classes/class_to_id.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/region_features_updated_masks/train','/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/color_features_single_classes_binary_updated_masks','train.json')
#load_features('/scratch/bcgp/datasets/visual_genome/vaw_dataset/color_data_single_classes/val.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/color_data_single_classes/class_to_id.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/region_features/dinov2/val','/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/color_features_single_classes','val.json')
load_features('/scratch/bcgp/datasets/visual_genome/vaw_dataset/data/test.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/color_data_single_classes/class_to_id.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/region_features_updated_masks/test','/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/color_features_single_classes_binary_updated_masks','test.json')

load_features('/scratch/bcgp/datasets/visual_genome/vaw_dataset/data/train.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/material_data_single_classes/class_to_id.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/region_features_updated_masks/train','/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/material_features_single_classes_binary_updated_masks','train.json')
#load_features('/scratch/bcgp/datasets/visual_genome/vaw_dataset/material_data_single_classes/val.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/material_data_single_classes/class_to_id.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/region_features/dinov2/val','/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/material_features_single_classes','val.json')
load_features('/scratch/bcgp/datasets/visual_genome/vaw_dataset/data/test.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/material_data_single_classes/class_to_id.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/region_features_updated_masks/test','/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/material_features_single_classes_binary_updated_masks','test.json')

load_features('/scratch/bcgp/datasets/visual_genome/vaw_dataset/data/train.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/shape_data_single_classes/class_to_id.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/region_features_updated_masks/train','/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/shape_features_single_classes_binary_updated_masks','train.json')
#load_features('/scratch/bcgp/datasets/visual_genome/vaw_dataset/shape_data_single_classes/val.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/shape_data_single_classes/class_to_id.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/region_features/dinov2/val','/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/shape_features_single_classes','val.json')
load_features('/scratch/bcgp/datasets/visual_genome/vaw_dataset/data/test.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/shape_data_single_classes/class_to_id.json','/scratch/bcgp/datasets/visual_genome/vaw_dataset/region_features_updated_masks/test','/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/shape_features_single_classes_binary_updated_masks','test.json')