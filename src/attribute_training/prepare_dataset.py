"""
Preparations for VAW dataset including formatting annotations, segmenting and cropping images.
"""
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
import vis_utils
from vis_utils import mask_and_crop_image
from PIL import Image 
from pycocotools import mask as mask_utils
import skimage 
import pickle 
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
        json.dump(new_annotations,fwrite)
def count_samples_per_annotation(annotation_file,attribute_to_index):
    samples = np.zeros(620)
    annotations = open_file(annotation_file)
    attribute_to_id = open_file(attribute_to_index)
    
    for entry in tqdm(annotations):
        positive_attributes = entry['negative_attributes']
        for attribute in positive_attributes:
            attribute_id = attribute_to_id[attribute]
            samples[attribute_id]+=1 
    save_file('/scratch/bcgp/datasets/visual_genome/vaw_dataset/data/num_samples_per_attribute_neg_val.json',samples.tolist())
 
def create_gt_for_val(annotation_file,attribute_to_index,num_classes=2):
    """
    If num classes==2, then each attribute is either positive or negative 
    If num_classes==3, then by default attribute is unspecified and is a 2. Only specified attributes are negative 
    """
    gt = []
    annotations = open_file(annotation_file)
    attribute_to_id = open_file(attribute_to_index)
    for entry in tqdm(list(annotations.values())):
        # not explicitly labeled 
        # only choose ones where we have features generated
        instance_id = entry['instance_id']
        if os.path.exists(os.path.join('/scratch/bcgp/datasets/vaw_cropped/features/test',instance_id+'.pkl')):
            positive_attributes = entry['positive_attributes']
            negative_attributes = entry['negative_attributes']
            if num_classes ==3:
                sample_preds = np.zeros(620)
                sample_preds.fill(2)
                for n in negative_attributes:
                    attribute_id = attribute_to_id[n]

                    sample_preds[attribute_id] = 0
            else:
                sample_preds = np.zeros(620)
            for p in positive_attributes:
                attribute_id = attribute_to_id[p]
                sample_preds[attribute_id] = 1
            gt.append(sample_preds)
    gt = np.stack(gt)
    with open('/scratch/bcgp/datasets/visual_genome/vaw_dataset/gt/test_gt_three_classes.npy','wb+') as f:
        np.save(f,gt)
def make_mask_from_bbox(bbox:list,image):
    h,w = image.size
    x,y,w,h = bbox 
    mask_image = np.zeros((h,w))
    mask_image[x:x+w,y:y+h] = 1 
    mask_h, mask_w = mask_image.shape 
    if mask_h != h or mask_w != w:
        print(f'Mask shape:{mask_image.shape}')
        print(f'Image shape:{image.size}')
        raise ValueError('Bbox from mask is wrong size')
    return mask_image 
def make_mask_annotations(annotation_file:str,save_dir:str):
    image_dict = {}
    all_annotations = open_file(annotation_file)
    for entry in tqdm(list(all_annotations.values())):
        image_id = entry['image_id']
        image = Image.open(os.path.join('/scratch/bcgp/datasets/visual_genome/images',image_id+'.jpg'))
        h,w = image.size 
        instance_entry = {}
        if entry['image_id'] not in image_dict:
            image_dict[entry['image_id']] = []
        if entry['instance_polygon'] != None:
            polygon = entry['instance_polygon'][0]
            mask_from_polygon = skimage.draw.polygon2mask((h,w),polygon)
            converted_mask = np.asfortranarray(mask_from_polygon).astype(np.uint8)
            rle_mask = mask_utils.encode(converted_mask)
        else:
            converted_mask = np.asfortranarray(make_mask_from_bbox(entry['instance_bbox'],image))
            rle_mask = mask_utils.encode(converted_mask.astype(np.uint8))
        instance_entry['instance_id'] = entry['instance_id']
        instance_entry['object_name'] = entry['object_name']
        instance_entry['positive_attributes'] = entry['positive_attributes']
        instance_entry['negative_attributes'] = entry['negative_attributes']
        instance_entry['segmentation'] = rle_mask 
        image_dict[image_id].append(instance_entry)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for img_regions in tqdm(image_dict,desc='Saving images'):
        regions = image_dict[img_regions]
        with open(os.path.join(save_dir,img_regions+'.pkl'),'wb') as f:
            pickle.dump(regions,f)
def check_new_annotations(old_attribute,new_colors,new_materials,new_shape):
    new_value = None 
    if old_attribute in new_colors:
        new_value = new_colors[old_attribute]
    elif old_attribute in new_materials:
        new_value = new_materials[old_attribute]
    else:
        new_value = new_shape[old_attribute]
    return new_value 
def revised_annotations(annotation_file:str,save_path:str,file_name:str):
    new_annotations = {}
    old_annotations = open_file(annotation_file)
    new_color_shape_material = open_file('subset_color_shape_material.json')
    all_colors = list(new_color_shape_material['color'].keys())
    all_materials = list(new_color_shape_material['material'].keys())
    all_shapes = list(new_color_shape_material['shape'].keys())
    all_keys = all_materials+all_colors+all_shapes
    for entry in tqdm(list(old_annotations.values())):
        new_instance_entry = {}
        positive_attributes = entry['positive_attributes']
        negative_attributes = entry['negative_attributes']
        new_positive_attributes = []
        new_negative_attributes = []
        add_to_dict = False 
        for p in positive_attributes:
            if p in all_keys:
                add_to_dict = True 
                new_attribute = check_new_annotations(p,new_color_shape_material['color'],new_color_shape_material['material'],new_color_shape_material['shape'])
                new_positive_attributes.append(new_attribute)
        for n in negative_attributes:
            if n in all_keys:
                add_to_dict = True 
                new_attribute = check_new_annotations(n,new_color_shape_material['color'],new_color_shape_material['material'],new_color_shape_material['shape'])
                new_negative_attributes.append(new_attribute)
        if add_to_dict:
            for k,v in list(entry.items()):
                new_instance_entry[k] = v 
            instance_id = new_instance_entry['instance_id']
            new_instance_entry['positive_attributes'] = positive_attributes
            new_instance_entry['negative_attributes'] = negative_attributes
            new_annotations[instance_id] = new_instance_entry
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file(os.path.join(save_path,file_name),new_annotations)
    print(len(list(new_annotations.keys())))
revised_annotations(annotation_file='/scratch/bcgp/datasets/visual_genome/vaw_dataset/data/test.json',save_path='/scratch/bcgp/datasets/visual_genome/vaw_dataset/smaller_num_classes',file_name='test_small_num_classes.json')
        




   
        

    


