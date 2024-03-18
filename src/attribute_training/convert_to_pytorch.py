import os 
import sys 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import json 
import torch 
import pickle 

from tqdm import tqdm 
import torch.nn as nn
import numpy as np 

def open_file(file_path:str,use_json=True):
    if use_json:
        with open(file_path) as fopen:
            data = json.load(fopen)
    else:
        with open(file_path,'rb') as fopen:
            data = pickle.load(fopen)
    return data 

color_classes = open_file('/scratch/bcgp/datasets/visual_genome/vaw_dataset/color_data_single_classes/id_to_class.json')
material_class = open_file('/scratch/bcgp/datasets/visual_genome/vaw_dataset/material_data_single_classes/id_to_class.json')
shape_class = open_file('/scratch/bcgp/datasets/visual_genome/vaw_dataset/shape_data_single_classes/id_to_class.json')

total = len(list(color_classes.keys()))+len(list(material_class.keys()))+len(list(shape_class.keys()))
att_dict = {'color':color_classes,'material':material_class,'shape':shape_class}
folder_dict = {'color':'dino_classifiers/binary/color','material':'dino_classifiers/binary/material','shape':'dino_classifiers/binary/shape'}
classifiers = nn.ModuleList([nn.Linear(768,out_features=1, bias=True) for _ in range(total+1)])
new_class_id_to_key = {}
current_key = 0 
for att in tqdm(att_dict):
    dictionary = att_dict[att]
    for class_id,class_name in dictionary.items():
        class_id = int(class_id)
        f = folder_dict[att]
        classifier= open_file(os.path.join('/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training',f,f'classifier_{str(class_id)}.pkl'),use_json=False)
        classifier.coef_ = classifier.coef_.astype(np.float32)
        classifier.intercept_ = classifier.intercept_.astype(np.float32)
        classifiers[int(current_key)].weight.data = torch.from_numpy(classifier.coef_)
        classifiers[int(current_key)].bias.data = torch.from_numpy(classifier.intercept_)
        new_class_id_to_key[current_key] = class_name
        current_key+=1 
torch.save(classifiers,'/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/all_dino_classifiers.pth',)
with open('/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/dino_class_id_to_index.json','w+') as f:
    json.dump(new_class_id_to_key,f)


