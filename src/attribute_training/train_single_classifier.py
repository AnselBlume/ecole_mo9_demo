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

def initialize_classifiers(attribute_name, clip_model, args):
    classifier = nn.Linear(args.enc_dim, 1, bias=False).to(args.device)
    attribute_vec = clip.tokenize([attribute_name])
    attribute_vec = attribute_vec.to(args.device)
    weight = clip_model.encode_text(attribute_vec).float()
    classifier.weight.data = weight 
    logging.info('Initialized clip weights')
    return classifier 
def eval_classifier(classifier,dataloader,weights_for_loss,args):
    # compute validation loss 
    total_loss = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            image = batch["image"].to(args.device)
            labels = batch["label"].to(args.device )
 
            # we get the union of the positive and negative attributes and they should be weighted more heavily compared to other attributes
            weight = torch.tensor(weights_for_loss).repeat(image.shape[0], 1).squeeze(-1).to(args.device)
            preds = classifier(image).squeeze(-1).squeeze(-1)
            logging.info(preds)
            # weight here is used to incorporate with negative attributes which is a crucial annotation
            loss = F.binary_cross_entropy_with_logits(preds, labels.float(), weight=weight)
            total_loss.append(loss.item())
    return np.mean(total_loss)





def train_classifier(classifier, train_dataloader, weights_for_loss, args,val_dataloader):
    optim = opt.AdamW(classifier.parameters(), lr=args.lr)
    previous_loss = 100000
    loss_did_not_decrease = 0
    for i,epoch in enumerate(range(args.n_epochs)):
        average_loss = []
        logging.info(f'Epoch:{i}')
        for batch in tqdm(train_dataloader):
            image = batch["image"].to(args.device)
            labels = batch["label"].to(args.device )
 
            # we get the union of the positive and negative attributes and they should be weighted more heavily compared to other attributes
            weight = torch.tensor(weights_for_loss).repeat(image.shape[0], 1).squeeze(-1).to(args.device)
            optim.zero_grad()
            preds = classifier(image).squeeze(-1).squeeze(-1)
            # weight here is used to incorporate with negative attributes which is a crucial annotation
            loss = F.binary_cross_entropy_with_logits(preds, labels.float(), weight=weight)
            loss.backward()
            optim.step()
            average_loss.append(loss.item())
        epoch_loss =np.mean(average_loss)
        wandb.log({"train_loss":epoch_loss})
        if epoch_loss<previous_loss:
            previous_loss = epoch_loss 
        else:
            loss_did_not_decrease+=1 
        
        val_loss  = eval_classifier(classifier,val_dataloader,weights_for_loss,args)
        wandb.log({"val_loss":val_loss})


        if i%args.log_every ==0:
            logging.info(f'Train Loss: {np.mean(average_loss)}')
            logging.info(f'Val Loss:{val_loss}')
            save_path = args.save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            torch.save(classifier, osp.join(save_path, f"classifier_{args.class_index}.pth"))

        if loss_did_not_decrease>args.stop_after:
            # stop if loss is not decreasing 
            break 

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
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--enc_dim", type=int, default=768)
    parser.add_argument("--backbone", type=str, default="ViT-L/14")
    parser.add_argument("--save_path", type=str, default="/scratch/bcgp/michal5/ecole_mo9_demo/classifiers")
    parser.add_argument("--weight_for_positive", type=float, default=1.0)
    parser.add_argument("--weight_for_negative", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--stop_after", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default='/scratch/bcgp/datasets')
    parser.add_argument("--class_id",type=int,help="value from 0 to 619")
    args = parser.parse_args()
    wandb.init(project=f'{args.class_id}_attribute_training', config=args)
    train_dataset = VAW(index_to_attribute=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/index_to_attribute.json'),
                    attribute_to_index=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/attribute_index.json'),
                    annotation_dir=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data'),
                    feature_dir=os.path.join(args.data_dir,'vaw_cropped/features'),
                    sample_count_file=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/num_samples_per_attribute_train.json'),
                    class_index=args.class_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataset = VAW(index_to_attribute=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/index_to_attribute.json'),
                    attribute_to_index=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/attribute_index.json'),
                    annotation_dir=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data'),
                    feature_dir=os.path.join(args.data_dir,'vaw_cropped/features'),
                    sample_count_file=os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/num_samples_per_attribute_val.json'),
                    split='val',
                    class_index=args.class_id)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
    id_to_attribute = train_dataset.id_to_attribute
    attribute_to_id = train_dataset.attribute_to_id 
    num_samples_per_attribute = train_dataset.num_samples_per_attribute
    attribute_name = id_to_attribute[str(args.class_id)]
    attribute_id = attribute_to_id[attribute_name]
    weights_for_loss = 1.0/num_samples_per_attribute[attribute_id]
    clip_model = clip.load(args.backbone, jit=False, device=args.device)[0].eval().requires_grad_(False)
    classifier = initialize_classifiers(attribute_name, clip_model, args)
    classifer = train_classifier(classifier, train_loader, weights_for_loss, args,val_loader)

    # save the classifiers
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    torch.save(classifier, osp.join(save_path, f"classifier_{args.class_index}.pth"))


