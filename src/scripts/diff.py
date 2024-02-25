# %%
import sys
import os
sys.path.append('/shared/nas2/michal5/ecole_mo9_demo/src')
sys.path.append('/shared/nas2/michal5/ecole_mo9_demo/src/image_processing')
from feature_extraction import CLIPFeatureExtractor, TrainedCLIPAttributePredictor, CLIPAttributePredictor
import torch
import pandas as pd 
import json
from tqdm import tqdm
from model.concept import ConceptKB
from kb_ops.train import ConceptKBTrainer
from feature_extraction import build_feature_extractor
from model.concept_predictor import ConceptPredictorOutput
from kb_ops.dataset import PresegmentedDataset, list_collate
from kb_ops.train_test_split import split_from_directory
from torch.utils.data import DataLoader
import numpy as np
import itertools 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL.Image import Image
from vis_utils import image_from_masks
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
import jsonargparse as argparse
from torchmetrics import Accuracy

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action=argparse.ActionConfigFile)

    parser.add_argument('--presegmented_dir',
                        default='/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/xiaomeng_augmented_data_segmentations',
                        help='Path to directory of preprocessed segmentations')

    parser.add_argument('--output_dir',
                        default='/shared/nas2/blume5/fa23/ecole/results/contribution_visualizations',
                        help='Path to output directory')

    parser.add_argument('--ckpt_path',
                        default='/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_02_20-08:02:42_abs_img_scale/concept_kb_epoch_15.pt',
                        help='Path to model checkpoint')

    return parser

def prediction_contributions(output: ConceptPredictorOutput, trained_attrs: list[str] = [], zs_attrs: list[str] = []):
    ret_dict = {}

    # Cumulative feature scores
    total_scores = {}
    total_scores['Image Features'] = output.img_score
    total_scores['Region Features'] = output.region_score
    total_scores['Trained Attributes: Image'] = output.trained_attr_img_scores.sum()
    total_scores['Trained Attributes: Regions'] = output.trained_attr_region_scores.sum()
    total_scores['Zero-shot Attributes: Image'] = output.zs_attr_img_scores.sum()
    total_scores['Zero-shot Attributes: Regions'] = output.zs_attr_region_scores.sum()

    total_mass = sum(map(lambda v: v.abs(), total_scores.values()))
    total_scores = {k : v.sgn() * v.abs() / total_mass for k, v in total_scores.items()}

    ret_dict['total_scores'] = total_scores

    # Trained attribute contributions
    if trained_attrs != []:
        t_attr_img_scores = {trained_attrs[i] : v for i, v in enumerate(output.trained_attr_img_scores)}
        t_attr_img_mass = output.trained_attr_img_scores.abs().sum()
        t_attr_img_scores = {k : v / t_attr_img_mass for k, v in t_attr_img_scores.items()}

        t_attr_region_scores = {trained_attrs[i] : v.sum() for i, v in enumerate(output.trained_attr_region_scores.T)}
        t_attr_region_mass = output.trained_attr_region_scores.abs().sum()
        t_attr_region_scores = {k : v / t_attr_region_mass for k, v in t_attr_region_scores.items()}

        ret_dict['trained_attr_img_scores'] = t_attr_img_scores
        ret_dict['trained_attr_region_scores'] = t_attr_region_scores

    # Zero-shot attribute contributions
    if zs_attrs != []:
        zs_attr_img_scores = {zs_attrs[i] : v for i, v in enumerate(output.zs_attr_img_scores)}
        zs_attr_img_mass = output.zs_attr_img_scores.abs().sum()
        zs_attr_img_scores = {k : v.sgn() * v.abs() / zs_attr_img_mass for k, v in zs_attr_img_scores.items()}

        zs_attr_region_scores = {zs_attrs[i] : v.sum() for i, v in enumerate(output.zs_attr_region_scores.T)}
        zs_attr_region_mass = output.zs_attr_region_scores.abs().sum()
        zs_attr_region_scores = {k : v.sgn() * v.abs() / zs_attr_region_mass for k, v in zs_attr_region_scores.items()}

        ret_dict['zs_attr_img_scores'] = zs_attr_img_scores
        ret_dict['zs_attr_region_scores'] = zs_attr_region_scores

    def tensors_to_nums(d: dict):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.item()

            elif isinstance(v, dict):
                d[k] = tensors_to_nums(v)

            elif isinstance(v, list):
                d[k] = [tensors_to_nums(x) for x in v]

        return d

    ret_dict = tensors_to_nums(ret_dict)

    return ret_dict

def visualize_prediction_contributions(img: Image, region_img: Image, prediction_contribs: dict, figsize=(15,8)):
    def plot_signed_hbar(ax, data: dict, top_k: int = None):
        labels = np.array(list(reversed(data.keys())))
        scores = np.array(list(reversed(data.values())))

        if top_k is not None:
            perm = np.argsort(np.abs(scores)) # Increasing order
            labels = labels[perm][-top_k:]
            scores = scores[perm][-top_k:]

        colors = ['red' if s < 0 else 'blue' for s in scores]
        ax.barh(labels, scores, color=colors)

    has_img = any('img_scores' in k for k in prediction_contribs.keys())
    has_zs = any('zs_attr' in k for k in prediction_contribs.keys())

    n_rows = 1 + has_img
    n_cols = 3 + has_zs # Image, total, trained attrs

    # Total scores
    fig = plt.figure(figsize=figsize, constrained_layout=True) # See https://stackoverflow.com/a/53642319
    gs = GridSpec(nrows=n_rows, ncols=n_cols, figure=fig, hspace=.08, wspace=.08)

    # Build region row
    ax = fig.add_subplot(gs[1 if has_img else 0, 0])
    ax.imshow(region_img)
    ax.axis('off')
    ax.set_title('Regions')

    ax = fig.add_subplot(gs[0:2 if has_img else 0, 1])
    plot_signed_hbar(ax, prediction_contribs['total_scores'])
    ax.set_title('Total Scores')

    ax = fig.add_subplot(gs[1 if has_img else 0, 2])
    plot_signed_hbar(ax, prediction_contribs['trained_attr_region_scores'], top_k=5)
    ax.set_title('Trained Attributes: Regions')

    if has_zs:
        ax = fig.add_subplot(gs[1 if has_img else 0, 3])
        plot_signed_hbar(ax, prediction_contribs['zs_attr_region_scores'], top_k=5)
        ax.set_title('Zero-shot Attributes: Regions')

    if has_img:
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(img)
        ax.axis('off')
        ax.set_title('Image')

        ax = fig.add_subplot(gs[0, 2])
        plot_signed_hbar(ax, prediction_contribs['trained_attr_img_scores'], top_k=5)
        ax.set_title('Trained Attributes: Image')

        if has_zs:
            ax = fig.add_subplot(gs[0, 3])
            plot_signed_hbar(ax, prediction_contribs['zs_attr_img_scores'], top_k=5)
            ax.set_title('Zero-shot Attributes: Image')

    return fig

def get_dataloader(dataset):
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=3, pin_memory=True, collate_fn=list_collate)
def diff(image_1,image_2,output_dir,top_k=5,figsize=(15,8)):
    image_1, image_1_attributes,image_1_name = image_1 
    image_1.save(image_1_name+'.jpg')
    image_2 , image_2_attributes,image_2_name = image_2 
    image_2.save(image_2_name+'.jpg')
    all_differences = {}
    for attribute in image_1_attributes:
        attribute_diff = abs(image_1_attributes[attribute]-image_2_attributes[attribute])
        all_differences[attribute]= attribute_diff 
     # get most different attributes 
    sorted_all_difference = [k for k, v in sorted(all_differences.items(), key=lambda item: item[1],reverse=True)]
    image_1_top_values = [image_1_attributes[k] for k in sorted_all_difference]
    image_2_top_values = [image_2_attributes[k] for k in sorted_all_difference]
    ind = np.arange(top_k)
    width = 0.4 
    fig, ax = plt.subplots()
    ax.barh(ind+width,image_1_top_values[:top_k],width,color='blue',label=image_1_name)
    
    ax.barh(ind,image_2_top_values[:top_k],width,color='orange',label=image_2_name)
    ax.set(yticks=ind + width, yticklabels=sorted_all_difference[:top_k],
    ylim=[2*width - 1, len(ind)])

    ax.legend(loc='upper right')
    save_path = os.path.join(output_dir,f'{image_1_name}_{image_2_name}.jpg')
    fig.savefig(save_path, bbox_inches='tight')
    
   
    
    

# %%
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # %%
    kb = ConceptKB.load(args.ckpt_path)
    feature_extractor = build_feature_extractor()
    trainer = ConceptKBTrainer(kb, feature_extractor)
    trained_attrs = feature_extractor.trained_clip_attr_predictor.attr_names
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir,exist_ok=True)
    # %%  Build datasets
    (trn_p, trn_l), (val_p, val_l), (tst_p, tst_l) = split_from_directory(args.presegmented_dir, exts='.pkl')

    test_ds = PresegmentedDataset(tst_p, tst_l)
    test_dl = get_dataloader(test_ds)
    all_attributes = []
    with torch.no_grad():
        for i,entry in enumerate(tqdm(test_dl,desc='predicting attributes')):
            instance = test_ds[i]
            image_path =  os.path.splitext(os.path.basename(instance['segmentations']['image_path']))[0]
            features = feature_extractor.forward(entry['segmentations'][0]['image'],[],[])
            attributes = feature_extractor.trained_clip_attr_predictor.predictor(features.image_features).sigmoid().squeeze()
            entry_attribute_dict = {att_name:att_value.item() for (att_name,att_value) in zip(trained_attrs,attributes)}
            entry_attributes = (entry['segmentations'][0]['image'],entry_attribute_dict,image_path)
            all_attributes.append(entry_attributes)
    combinations = itertools.combinations(all_attributes,2)
    all_combos = [c for c in combinations]
    for pair in all_combos:
        attr_1, attr_2 = pair 
        diff(attr_1,attr_2,args.output_dir)
   