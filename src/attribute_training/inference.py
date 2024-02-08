import os 
import sys 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from vaw_dataset import VAW 
from tqdm import tqdm 
import logging 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_classifiers(args):
    saved_models = torch.load(args.save_path)
    classifiers = nn.ModuleList([nn.Linear(args.enc_dim, 1, bias=False) for _ in range(len(saved_models))]).to(args.device)
    
    for i in range(len(saved_models)):
        classifiers[i].weight = saved_models[i].weight 
    classifiers = classifiers.to(args.device)
    return classifiers

def infer(image_input,index_to_attribute,attribute_to_index,attribute=None,threshold=0.5):
    if attribute == None:
        preds =  torch.cat([classifier(image) for classifier in classifiers], dim=1).squeeze(-1)
        index = preds[preds>threshold]
        for i in index:
            print(f'Attribute:{index_to_attribute[i]} with confidence {preds[i]}')

