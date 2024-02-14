import os 
import sys 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import clip
import torch
import torch.optim as opt
import numpy as np
import torch.nn as nn
import logging 
from tqdm import tqdm 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import os.path as osp
from vaw_dataset import VAW 
import argparse
import wandb 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def initialize_classifiers(attribute_name, clip_model, args):
    classifier = nn.Linear(args.enc_dim, 1, bias=False).to(args.device)
    logit_scale = torch.ones([])*np.log(1/.07)
    attribute_vec = clip.tokenize([attribute_name]).to(args.device)
    weight = clip_model.encode_text(attribute_vec).float()
    weight /=weight.norm(dim=-1,keepdim=True)
    weight*=logit_scale.exp()
    classifier.weight.data = weight
    logger.info('Initialized clip weights')
    return classifier 
def eval_classifier(classifier,dataloader,args):
    # compute validation loss 
    classifier.eval()
    total_loss = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            #batch_size,_ = batch.size()

            image = batch["image"].squeeze(1).to(args.device)
            image = image/image.norm(dim=1,keepdim=True)
        
            positive = batch["positive"].to(args.device)
            # we currently do not consider negative attributes
            negative = batch["negative"].to(args.device)
            # labels = torch.cat([positive, negative], dim=0)
            no_label = batch["unknown"].to(device=args.device,dtype=torch.bool)
            bs = positive.size()[0]


            #total_no_label = torch.count_nonzero(no_label)
            actual_labels =torch.count_nonzero(no_label)
            #TODO: Compute loss without unknown labels
            labels = positive
            pred_output = []
     
            preds= classifier(image)
            
            #preds =  torch.cat([p for p in pred_output], dim=1)
            # weight here is used to incorporate with negative attributes which is a crucial annotation
            total_count= torch.count_nonzero(no_label)
            if total_count >0:
                loss = F.binary_cross_entropy_with_logits(preds.squeeze()[no_label], labels.float()[no_label],reduction='mean')

                #actual_loss = torch.divide(loss.sum(),(actual_labels+1e-10))
                total_loss.append(loss.item())
            
    return np.mean(total_loss)





def train_classifier(classifier, train_dataloader, neg_samples_per_att,pos_samples_per_att, args,val_dataloader,pos_weight=None):

    optim = opt.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    previous_loss = 100000
    loss_did_not_decrease = 0
    for i,epoch in enumerate(range(args.n_epochs)):
        classifier.train()
        average_loss = []
        logging.info(f'Epoch:{i}')
        for batch in tqdm(train_dataloader):
            image = batch["image"].squeeze(1).to(args.device)
            image = image/image.norm(dim=1,keepdim=True)
                                
    

            positive = batch["positive"]
            #print(torch.numel(positive),positive.size()[0]*positive.size()[1])
            # we currently do not consider negative attributes
            negative = batch["negative"]
            # labels = torch.cat([positive, negative], dim=0)
            no_label = batch["unknown"].to(device=args.device,dtype=torch.bool) 
            labels = positive.to(args.device)
            # we get the union of the positive and negative attributes and they should be weighted more heavily compared to other attributes
            bs = positive.size()[0]
            # positive = positive.to(args.device)
            # negative = negative.to(args.device)
            #weights = weights.to(args.device)
            # weight[positive] *= args.weight_for_positive
            # weight[negative] *= args.weight_for_negative
            optim.zero_grad()
            preds = classifier(image)
            
            
            # weight here is used to incorporate with negative attributes which is a crucial annotation
            total_count= torch.count_nonzero(no_label)
            if total_count>0:
                loss = F.binary_cross_entropy_with_logits(preds.squeeze()[no_label], labels.float()[no_label],reduction='mean')
                #average_over_nonzero = torch.divide(loss.sum(),(total_count+1e-10))
                #average_over_nonzero.backward()
                loss.backward()
                optim.step()
                average_loss.append(loss.item())
        epoch_loss =np.mean(average_loss)
 
        val_loss  = eval_classifier(classifier,val_dataloader,args)
        if abs(val_loss-previous_loss)<1e-4:
            loss_did_not_decrease+=1
        elif val_loss<previous_loss:
            previous_loss = val_loss 
        else:
            loss_did_not_decrease+=1 

        wandb.log({"val_loss":val_loss})
        wandb.log({"train_loss":epoch_loss})
        logger.info(f'Train Loss: {np.mean(average_loss)}')
        logger.info(f'Val Loss:{val_loss}')

        if i%args.save_every ==0:
            save_path = args.save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            torch.save(classifier, osp.join(save_path, f"classifiers_{str(args.class_id)}.pth"))

        if loss_did_not_decrease>args.stop_after:
            # stop if loss is not decreasing 
            break 
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    torch.save(classifier, osp.join(save_path, f"classifiers_{str(args.class_id)}.pth"))

    return classifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--enc_dim", type=int, default=768)
    parser.add_argument("--backbone", type=str, default="ViT-L/14")
    parser.add_argument("--save_path", type=str, default="/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/classifiers_fix")
    parser.add_argument("--weight_for_positive", type=float, default=1.0)
    parser.add_argument("--weight_for_negative", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--stop_after", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default='/scratch/bcgp/datasets')
    parser.add_argument("--class_id",type=int,help="value from 0 to 619",default=0)
    parser.add_argument("--wandb_project_name",type=str,default='attribute_training')
    parser.add_argument("--wandb_run_name",type=str,default='class_id')
    
    
    args = parser.parse_args()
    if args.wandb_run_name == 'class_id':
        wandb_run_name = f'class_id_{str(args.class_id)}'
    else:
        wandb_run_name = args.wandb_run_name 
    wandb.init(project=args.wandb_project_name, config=args,name=wandb_run_name)
    train_dataset = VAW(index_to_attribute=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/index_to_attribute.json'),
                    attribute_to_index=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/attribute_index.json'),
                    annotation_dir=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data'),
                    feature_dir=os.path.join(args.data_dir,'vaw_cropped/features'),
                    sample_count_file=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/num_samples_per_attribute_train.json'),
                    sample_neg_count_file=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/num_samples_per_attribute_neg_train.json'),class_index=args.class_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataset = VAW(index_to_attribute=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/index_to_attribute.json'),
                    attribute_to_index=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/attribute_index.json'),
                    annotation_dir=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data'),
                    feature_dir=os.path.join(args.data_dir,'vaw_cropped/features'),
                    sample_count_file=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/num_samples_per_attribute_val.json'),
                    sample_neg_count_file=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/num_samples_per_attribute_neg_val.json'),
                    split='val',class_index=args.class_id)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)    
    id_to_attribute = train_dataset.id_to_attribute
    attribute_to_id = train_dataset.attribute_to_id 
    num_samples_per_pos_attribute = train_dataset.num_pos_samples_per_attribute
    attribute_dict = train_dataset.id_to_attribute
    num_samples_per_attribute = train_dataset.num_pos_samples_per_attribute
    num_samples_per_neg_attribute = train_dataset.num_neg_samples_per_attribute
    pos_weight = np.array([1.0 / num_samples_per_attribute[int(attribute)] if num_samples_per_attribute[int(attribute)] > 0 else 0.0 for attribute in attribute_dict])
    neg_weight =  np.array([1.0 / num_samples_per_neg_attribute[int(attribute)] if num_samples_per_neg_attribute[int(attribute)] > 0 else 0.0 for attribute in attribute_dict])
    
    clip_model = clip.load(args.backbone, jit=False, device=args.device)[0].eval().requires_grad_(False)

    classifiers = initialize_classifiers(id_to_attribute[str(args.class_id)], clip_model, args)
    classifiers = train_classifier(classifiers, train_loader, torch.from_numpy(neg_weight),torch.from_numpy(pos_weight), args,val_loader)

    # save the classifiers
    save_path = args.save_path
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)
    torch.save(classifiers.cpu(), osp.join(save_path, f"classifiers_{str(args.class_id)}.pth"))

