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
    classifiers = nn.ModuleList([nn.Linear(args.enc_dim, 1, bias=True) for _ in range(len(saved_models))]).to(args.device)
    
    for i in range(len(saved_models)):
        classifiers[i].weight = saved_models[i].weight 
        classifiers[i].bias = saved_models[i].bias
    classifiers = classifiers.to(args.device)
    return classifiers 

def make_predictions(classifiers,dataloader,args):
    """
    Make predictions and save into npy file 
    """
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            image = batch["image"].to(args.device)
            positive = batch["positive"].to(args.device)
            # we currently do not consider negative attributes
            negative = batch["negative"].to(args.device)
            # labels = torch.cat([positive, negative], dim=0)
            labels = positive
            preds =  torch.cat([classifier(image) for classifier in classifiers], dim=1).squeeze(-1)
            predictions.append(preds.cpu().numpy())
    predictions = np.stack(predictions)
    with open(args.predict_file,'wb+') as f:
        np.save(f,predictions)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--predict_file", type=str)
    parser.add_argument("--device", type=str,default='cuda')
    parser.add_argument("--batch_size", type=int,default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--enc_dim", type=int, default=768)
    parser.add_argument("--data_dir", type=str,default='/scratch/bcgp/datasets')
    args = parser.parse_args([
        '--save_path','/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/classifiers_bias/classifiers.pth',
        '--predict_file','/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/classifiers/test_two_class_preds.npy'
    ])
    test_dataset = VAW(index_to_attribute=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/index_to_attribute.json'),
                    attribute_to_index=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/attribute_index.json'),
                    annotation_dir=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data'),
                    feature_dir=os.path.join(args.data_dir,'vaw_cropped/features'),
                    sample_neg_count_file=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/num_samples_per_attribute_neg_val.json'),
                    sample_count_file=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/num_samples_per_attribute_val.json'),
                    split='test')
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)

    classifiers = load_classifiers(args)
    make_predictions(classifiers,test_loader,args)


