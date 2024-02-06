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
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import os.path as osp
import argparse
from vaw_dataset import VAW 
import wandb 
from tqdm import tqdm 
import logging 
def initialize_classifiers(attribute_dict, clip_model, args):
    num_attributes = len(attribute_dict)
    classifiers = nn.ModuleList([nn.Linear(args.enc_dim, 1, bias=False) for _ in range(num_attributes)])
    # This is to use the text_encoder from CLIP to encode the attribute names as initial weights for the classifiers
    for attribute, classifier in zip(attribute_dict, classifiers):
        attribute_vec = clip.tokenize([attribute])
        weight = clip_model.encode_text(attribute_vec).float()
        classifier.weight.data = weight
    classifiers = classifiers.to(args.device)
    logging.info('Initialized CLIP classifiers')
    return classifiers
def eval_classifier(classifier,dataloader,weights_for_loss,args):
    # compute validation loss 
    total_loss = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            image = batch["image"].to(args.device)
            positive = batch["positive"].to(args.device)
            # we currently do not consider negative attributes
            negative = batch["negative"].to(args.device)
            # labels = torch.cat([positive, negative], dim=0)
            labels = positive
 
            # we get the union of the positive and negative attributes and they should be weighted more heavily compared to other attributes
            weight = torch.tensor(weights_for_loss).repeat(image.shape[0], 1).to(args.device)

            preds =  torch.cat([classifier(image) for classifier in classifiers], dim=1)
            # weight here is used to incorporate with negative attributes which is a crucial annotation
            loss = F.binary_cross_entropy_with_logits(preds, labels.float(), weight=weight)
            total_loss.append(loss.item())
    return np.mean(total_loss)
def train_classifier(classifiers, train_dataloader, weights_for_loss, args,val_dataloader):
    optim = opt.Adam(classifiers.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    previous_loss = 100000
    loss_did_not_decrease = 0
    for epoch in enumerate(range(args.n_epochs)):
        average_loss = []
        logging.info(f'Epoch:{i}')
        for batch in tqdm(train_dataloader):
            image = batch["image"].to(args.device)
            positive = batch["positive"].to(args.device)
            # we currently do not consider negative attributes
            negative = batch["negative"].to(args.device)
            # labels = torch.cat([positive, negative], dim=0)
            labels = positive
            # we get the union of the positive and negative attributes and they should be weighted more heavily compared to other attributes
            weight = torch.tensor(weights_for_loss).repeat(image.shape[0], 1).to(args.device)
            weight[positive] *= args.weight_for_positive
            weight[negative] *= args.weight_for_negative
            optim.zero_grad()
            preds = torch.cat([classifier(image) for classifier in classifiers], dim=1)
            # weight here is used to incorporate with negative attributes which is a crucial annotation
            loss = F.binary_cross_entropy_with_logits(preds, labels.float(), weight=weight)
            loss.backward()
            optim.step()
            average_loss.append(loss.item())
        epoch_loss =np.mean(average_loss)
        
        if epoch_loss<previous_loss:
            previous_loss = epoch_loss 
        else:
            loss_did_not_decrease+=1 
        val_loss  = eval_classifier(classifier,val_dataloader,weights_for_loss,args)
        if args.use_wandb:
            wandb.log({"val_loss":val_loss})
            wandb.log({"epoch_loss":epoch_loss})
        if i%args.log_every ==0:
            logging.info(f'Train Loss: {np.mean(average_loss)}')
            logging.info(f'Val Loss:{val_loss}')
            save_path = args.save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            torch.save(classifiers, osp.join(save_path, f"classifier_{args.class_index}.pth"))

        if loss_did_not_decrease>args.stop_after:
            # stop if loss is not decreasing 
            break 

    return classifiers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--enc_dim", type=int, default=768)
    parser.add_argument("--backbone", type=str, default="ViT-L/14")
    parser.add_argument("--save_path", type=str, default="path/to/save")
    parser.add_argument("--weight_for_positive", type=float, default=2.0)
    parser.add_argument("--weight_for_negative", type=float, default=2.0)
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--stop_after", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default='/scratch/bcgp/datasets')

    args = parser.parse_args()

    # data loading: should be a class of dataset and then dataloader.
    # When sampling from the dataset, it produces {"image": a feature vector, "positive": one hot vector where 1s indicate
    # the image contains those attributes, "negative": one hot vector}
    # We should be able to get how many attributes and the corresponding names from the dataset: dataset.id_to_attribute
    # which is a dictionary of id to attribute name
    if args.use_wandb:
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
    
    attribute_dict = dataset.id_to_attribute
    num_samples_per_attribute = dataset.num_samples_per_attribute
    weights_for_loss = np.array([1.0 / num_samples_per_attribute[attribute_to_id[attribute]] if num_samples_per_attribute[attribute_to_id[attribute]] > 0 else 0.0 for attribute in attribute_dict])

    clip_model = clip.load(args.backbone, jit=False, device=args.device)[0].eval().requires_grad_(False)

    classifiers = initialize_classifiers(list(id_to_attribute.values()), clip_model, args)
    classifiers = train_classifier(classifiers, train_loader, weights_for_loss, args,val_loader)

    # save the classifiers
    save_path = args.save_path
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)
    torch.save(classifiers, osp.join(save_path, "classifiers.pth"))