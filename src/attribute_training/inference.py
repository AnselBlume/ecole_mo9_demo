import os 
import sys 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import numpy as np 
import torch 
import torch.nn as nn
import clip 
import pickle 
import json 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from vaw_dataset import VAW 
from tqdm import tqdm 
import logging 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def initialize_single_classifier(attribute_name,clip_model,args):
    logit_scale = torch.ones([])*np.log(1/.07)
    #classifiers = nn.ModuleList([nn.Linear(args.enc_dim, 1, bias=False) for _ in range(num_attributes)]).to(args.device)
    # This is to use the text_encoder from CLIP to encode the attribute names as initial weights for the classifiers
    classifier = nn.Linear(768,1,bias=False)
    attribute_vec = clip.tokenize([attribute_name]).to(args.device)
    weight = clip_model.encode_text(attribute_vec).float()
    weight/=weight.norm(dim=-1,keepdim=True)
    weight*=logit_scale.exp()
    classifier.weight.data = weight
    classifier.eval()
    return classifier 

def load_classifiers(class_ids,zeroshot,id_to_attribute, clip_model, args):
    classifier_list = []
    if zeroshot:
        print('hello')
        for attribute_name in list(id_to_attribute.values()):
            classifier = initialize_single_classifier(attribute_name,clip_model,args)
            classifier_list.append(classifier)
    else:
        for i in list(id_to_attribute.keys()):
            path_name = os.path.join(args.model_dir,f'classifiers_{i}.pth')
            if i in class_ids and os.path.exists(path_name):
                classifier = torch.load(path_name).to(args.device)
                classifier.eval()
                classifier_list.append(classifier)
            else:
                attribute_value = id_to_attribute[i]
                classifier = initialize_single_classifier(attribute_value,clip_model,args)
                classifier_list.append(classifier)
    return classifier_list 

def get_actual_id(top_ids,class_ids):
    actual_ids = []
    for top_id in top_ids:
        actual_ids.append(class_ids[top_id])
    return actual_ids

def top_preds(classifier_output,class_ids,top_k=1):
    selected_classes = classifier_output[class_ids]
    descending_idx = selected_classes.argsort()[::-1]
    descending_values = selected_classes[descending_idx]
    actual_ids =get_actual_id(descending_idx[:top_k],class_ids) 
    return actual_ids, descending_values[:top_k]
def infer(args):
    print(args.zeroshot)
    with open(f'/scratch/bcgp/datasets/vaw_cropped/features/val/{args.instance_id}.pkl','rb') as f:
        image = pickle.load(f)
    image = torch.from_numpy(image).squeeze(1)
    image = image/image.norm(dim=1,keepdim=True).squeeze(1)
    image = image.to(args.device)
    with open('/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/id_to_color_shape_material.json','r+') as f:
        selected_classes = json.load(f)
    with open('/scratch/bcgp/datasets/visual_genome/vaw_dataset/data/index_to_attribute.json','r+') as f:
        id_to_attribute = json.load(f)
    class_ids = list(selected_classes.keys())
    clip_model = clip.load('ViT-L/14', jit=False, device=args.device)[0].eval().requires_grad_(False)

    classifier_list = load_classifiers(class_ids,args.zeroshot,id_to_attribute, clip_model, args)

    with torch.no_grad():
        preds =  torch.cat([torch.sigmoid(classifier(image)) for classifier in classifier_list], dim=1).squeeze(0).cpu().numpy()
    with open('/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/color_shape_material_dict_ids.json','r+') as f:
        color_shape_material_id = json.load(f)
    top_colors,top_color_values = top_preds(preds,color_shape_material_id['color'],top_k=args.top_k)
    logger.info('Top Color Predictions')
    for color_id, color_conf in zip(top_colors,top_color_values):
        print(f'Color:{id_to_attribute[str(color_id)]}, Conf:{color_conf}')

    top_material,top_material_values = top_preds(preds,color_shape_material_id['material'],top_k=args.top_k)
    logger.info('Top Material Predictions')

    for material_id, material_conf in zip(top_material,top_material_values):
        print(f'Material:{id_to_attribute[str(material_id)]}, Conf:{material_conf}')
   
    top_shape,top_shape_values = top_preds(preds,color_shape_material_id['shape'],top_k=args.top_k)
    logger.info('Top Shape Predictions')
    for shape_id, shape_conf in zip(top_shape,top_shape_values):
        print(f'Shape:{id_to_attribute[str(shape_id)]}, Conf:{shape_conf}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--device", type=str,default='cuda')
    parser.add_argument("--instance_id",type=str)
    parser.add_argument("--zeroshot",action='store_true')
    parser.add_argument("--top_k",type=int,default=5)
    
    args = parser.parse_args()

    infer(args)