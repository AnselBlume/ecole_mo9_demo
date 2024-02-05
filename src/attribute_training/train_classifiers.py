mport clip
import torch
import torch.optim as opt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import os.path as osp
import argparse

def initialize_classifiers(attribute_dict, clip_model, args):
    num_attributes = len(attribute_dict)
    classifiers = nn.ModuleList([nn.Linear(args.enc_dim, 1, bias=False) for _ in range(num_attributes)])
    # This is to use the text_encoder from CLIP to encode the attribute names as initial weights for the classifiers
    for attribute, classifier in zip(attribute_dict, classifiers):
        attribute_vec = clip.tokenize([attribute])
        weight = clip_model.encode_text(attribute_vec).float()
        classifier.weight.data = weight
    classifiers = classifiers.to(args.device)
    return classifiers

def train_classifier(classifiers, dataloader, weights_for_loss, args):
    optim = opt.Adam(classifiers.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # criterion = nn.BCEWithLogitsLoss()
    for epoch in range(args.n_epochs):
        for batch in dataloader:
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
    parser.add_argument("--enc_dim", type=int, default=512)
    parser.add_argument("--backbone", type=str, default="ViT-B/32")
    parser.add_argument("--save_path", type=str, default="path/to/save")
    parser.add_argument("--weight_for_positive", type=float, default=2.0)
    parser.add_argument("--weight_for_negative", type=float, default=2.0)
    args = parser.parse_args()

    # data loading: should be a class of dataset and then dataloader.
    # When sampling from the dataset, it produces {"image": a feature vector, "positive": one hot vector where 1s indicate
    # the image contains those attributes, "negative": one hot vector}
    # We should be able to get how many attributes and the corresponding names from the dataset: dataset.id_to_attribute
    # which is a dictionary of id to attribute name
    dataset = ...
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    attribute_dict = dataset.id_to_attribute
    num_samples_per_attribute = dataset.num_samples_per_attribute
    weights_for_loss = np.array([1.0 / num_samples_per_attribute[attribute] if num_samples_per_attribute[attribute] > 0 else 0.0 for attribute in attribute_dict])

    clip_model = clip.load(args.backbone, jit=False, device=args.device)[0].eval().requires_grad_(False)

    classifiers = initialize_classifiers(attribute_dict, clip_model, args)
    train_classifier(classifiers, dataloader, weights_for_loss, args)

    # save the classifiers
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    torch.save(classifiers, osp.join(save_path, "classifiers.pth"))