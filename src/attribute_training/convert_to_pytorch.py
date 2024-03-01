import os 
import sys 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import clip
import json 
import torch 
import pickle 
import sklearn 
from tqdm import tqdm 
import torch.nn as nn
import numpy as np 

def initialize_classifiers(attribute_name, clip_model):
    classifier = nn.Linear(768, 1, bias=True).double()
    logit_scale = torch.ones([])*np.log(1/.07)
    attribute_vec = clip.tokenize([attribute_name]).to('cuda')
    weight = clip_model.encode_text(attribute_vec).float()
    weight /=weight.norm(dim=-1,keepdim=True)
    weight*=logit_scale.exp()
    classifier.weight.data = weight
    return classifier.cpu().double() 

with open('/scratch/bcgp/datasets/visual_genome/vaw_dataset/data/index_to_attribute.json','r+') as fopen:
    index_to_attribute = json.load(fopen)
num_attributes = 619 
clip_model = clip.load("ViT-L/14", jit=False, device='cuda')[0].eval().requires_grad_(False)
classifiers = nn.ModuleList([nn.Linear(768,out_features=1, bias=True) for _ in range(num_attributes+1)])
for index, attribute_name in tqdm(index_to_attribute.items()):
    classifier_name = f'classifier_{index}.pkl'
    classifiers[int(index)] = classifiers[int(index)].double()
    if os.path.exists(f'sklearn_classifiers/{classifier_name}'):
        with open(f'sklearn_classifiers/{classifier_name}','rb') as fopen:
            sklearn_classifier = pickle.load(fopen)
            classifiers[int(index)].weight.data = torch.from_numpy(sklearn_classifier.coef_)
            classifiers[int(index)].bias.data = torch.from_numpy(sklearn_classifier.intercept_)
    else:
        classifiers[int(index)] = initialize_classifiers(attribute_name, clip_model)
torch.save(classifiers,'classifiers_official.pth')

