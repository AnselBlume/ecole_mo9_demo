# %%
import os

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
from itertools import chain

import jsonargparse as argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from feature_extraction import build_feature_extractor
from kb_ops import ConceptKBFeaturePipeline, ConceptKBPredictor
from kb_ops.dataset import PresegmentedDataset, list_collate
from kb_ops.train_test_split import split_from_paths
from matplotlib.gridspec import GridSpec
from model.concept import ConceptKB
from model.concept.concept_predictor import ConceptPredictorOutput
from PIL.Image import Image
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from tqdm import tqdm
from visualization.vis_utils import image_from_masks


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action=argparse.ActionConfigFile)

    parser.add_argument('--output_dir',
                        default='/shared/nas2/blume5/fa23/ecole/results/contribution_visualizations',
                        help='Path to output directory')

    parser.add_argument('--ckpt_path',
                        default='/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_02_22-00:47:36-s1roip9b-two_scales/concept_kb_epoch_15.pt',
                        help='Path to model checkpoint')

    return parser

def prediction_contributions(output: ConceptPredictorOutput, trained_attrs: list[str] = [], zs_attrs: list[str] = []):
    ret_dict = {}

    # Values may be None depending on whether we're using the full image or regions; this method helps reduce
    # lots of explicit if checks in the code
    def set_value_if_not_none(d: dict, dict_key: str, optional_tensor, sum_tensor: bool = False):
        if optional_tensor is not None:
            d[dict_key] = optional_tensor.sum() if sum_tensor else optional_tensor

    # Cumulative feature scores
    total_scores = {}
    set_value_if_not_none(total_scores, 'Image Features', output.img_score)
    set_value_if_not_none(total_scores, 'Region Features', output.region_score)

    if trained_attrs != []:
        set_value_if_not_none(total_scores, 'Trained Attributes: Image', output.trained_attr_img_scores, sum_tensor=True)
        set_value_if_not_none(total_scores, 'Trained Attributes: Regions', output.trained_attr_region_scores, sum_tensor=True)

    if zs_attrs != []:
        set_value_if_not_none(total_scores, 'Zero-shot Attributes: Image', output.zs_attr_img_scores, sum_tensor=True)
        set_value_if_not_none(total_scores, 'Zero-shot Attributes: Regions', output.zs_attr_region_scores, sum_tensor=True)

    total_mass = sum(map(lambda v: v.abs(), total_scores.values()))
    total_scores = {k : v / total_mass for k, v in total_scores.items()}

    ret_dict['total_scores'] = total_scores

    # Trained attribute contributions
    uses_full_img = output.img_score is not None
    uses_regions = output.region_score is not None

    if trained_attrs != []:
        if uses_full_img:
            t_attr_img_scores = {trained_attrs[i] : v for i, v in enumerate(output.trained_attr_img_scores)}
            t_attr_img_mass = output.trained_attr_img_scores.abs().sum()
            t_attr_img_scores = {k : v / t_attr_img_mass for k, v in t_attr_img_scores.items()}

            ret_dict['trained_attr_img_scores'] = t_attr_img_scores

        if uses_regions:
            t_attr_region_scores = {trained_attrs[i] : v.sum() for i, v in enumerate(output.trained_attr_region_scores.T)}
            t_attr_region_mass = output.trained_attr_region_scores.abs().sum()
            t_attr_region_scores = {k : v / t_attr_region_mass for k, v in t_attr_region_scores.items()}

            ret_dict['trained_attr_region_scores'] = t_attr_region_scores

    # Zero-shot attribute contributions
    if zs_attrs != []:
        if uses_full_img:
            zs_attr_img_scores = {zs_attrs[i] : v for i, v in enumerate(output.zs_attr_img_scores)}
            zs_attr_img_mass = output.zs_attr_img_scores.abs().sum()
            zs_attr_img_scores = {k : v.sgn() * v.abs() / zs_attr_img_mass for k, v in zs_attr_img_scores.items()}

            ret_dict['zs_attr_img_scores'] = zs_attr_img_scores

        if uses_regions:
            zs_attr_region_scores = {zs_attrs[i] : v.sum() for i, v in enumerate(output.zs_attr_region_scores.T)}
            zs_attr_region_mass = output.zs_attr_region_scores.abs().sum()
            zs_attr_region_scores = {k : v.sgn() * v.abs() / zs_attr_region_mass for k, v in zs_attr_region_scores.items()}

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

def visualize_prediction_contributions(
    img: Image,
    region_img: Image,
    prediction_contribs: dict,
    figsize=(15,8)
):
    def plot_signed_hbar(ax, data: dict, top_k: int = None):
        labels = np.array(list(reversed(data.keys())))
        scores = np.array(list(reversed(data.values())))

        if top_k is not None:
            perm = np.argsort(np.abs(scores)) # Increasing order
            labels = labels[perm][-top_k:]
            scores = scores[perm][-top_k:]

        colors = ['red' if s < 0 else 'blue' for s in scores]
        ax.barh(labels, scores, color=colors)

    uses_full_img = any('image' in k.lower() for k in prediction_contribs['total_scores'].keys())
    uses_regions = any('regions' in k.lower() for k in prediction_contribs['total_scores'].keys())
    has_zs = any('zs_attr' in k for k in prediction_contribs.keys())

    n_rows = 1 + (uses_full_img and uses_regions)
    n_cols = 3 + has_zs # Image, total, trained attrs

    # Total scores
    fig = plt.figure(figsize=figsize, constrained_layout=True) # See https://stackoverflow.com/a/53642319
    gs = GridSpec(nrows=n_rows, ncols=n_cols, figure=fig, hspace=.08, wspace=.08)

    if uses_full_img and uses_regions:
        ax = fig.add_subplot(gs[0:2, 1])
    else:
        ax = fig.add_subplot(gs[0, 1])
    plot_signed_hbar(ax, prediction_contribs['total_scores'])
    ax.set_title('Total Scores')

    # Build region row
    if uses_regions:
        ax = fig.add_subplot(gs[1 if uses_full_img else 0, 0])
        ax.imshow(region_img)
        ax.axis('off')
        ax.set_title('Regions')

        ax = fig.add_subplot(gs[1 if uses_full_img else 0, 2])
        plot_signed_hbar(ax, prediction_contribs['trained_attr_region_scores'], top_k=5)
        ax.set_title('Trained Attributes: Regions')

        if has_zs:
            ax = fig.add_subplot(gs[1 if uses_full_img else 0, 3])
            plot_signed_hbar(ax, prediction_contribs['zs_attr_region_scores'], top_k=5)
            ax.set_title('Zero-shot Attributes: Regions')

    if uses_full_img:
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
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=list_collate)

# %%
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    # args = parser.parse_args(['--ckpt_path', '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_03_06-06:00:17-goiur8to/concept_kb_epoch_15.pt'])

    # %%
    kb = ConceptKB.load(args.ckpt_path)
    feature_pipeline = ConceptKBFeaturePipeline(kb, None, build_feature_extractor())
    predictor = ConceptKBPredictor(kb, feature_pipeline)
    trained_attrs = feature_pipeline.feature_extractor.trained_attr_predictor.attr_names

    # %%  Build datasets
    all_segmentation_paths = list(chain.from_iterable([
        [ex.image_segmentations_path for ex in c.examples]
        for c in kb
    ]))
    (trn_p, trn_l), (val_p, val_l), (tst_p, tst_l) = split_from_paths(all_segmentation_paths)

    test_ds = PresegmentedDataset(tst_p, tst_l)
    test_dl = get_dataloader(test_ds)

    # %% Predict
    predictions = predictor.predict(test_dl)

    # %%
    accuracy = Accuracy(task='multiclass', num_classes=len(kb))

    all_contributions = []
    os.makedirs(args.output_dir, exist_ok=True)

    for index in tqdm(range(len(predictions)), desc='Visualizing contributions'):
        instance = test_ds[index]
        prediction = predictions[index]

        # Create images
        img = instance['segmentations'].input_image
        region_img = to_pil_image(
            image_from_masks(instance['segmentations'].part_masks, superimpose_on_image=pil_to_tensor(img), superimpose_alpha=.7)
        ) if instance['segmentations'].part_masks.shape[0] > 0 else img

        # Attribute scores
        predicted_concept_outputs = prediction['predicted_concept_outputs']
        predicted_concept_zs_attrs = [attr.name for attr in kb[prediction['predicted_label']].zs_attributes]
        contributions = prediction_contributions(predicted_concept_outputs, trained_attrs, predicted_concept_zs_attrs)
        all_contributions.append(contributions)

        # Create figure
        fig = visualize_prediction_contributions(img, region_img, contributions)

        # Update accuracy
        pred_ind = torch.tensor(prediction['predicted_index']).unsqueeze(0)
        true_ind = torch.tensor(prediction['true_index']).unsqueeze(0)
        accuracy(pred_ind, true_ind)

        # Set title
        pred_label = prediction['concept_names'][prediction['predicted_index']].capitalize()
        true_label = prediction['concept_names'][prediction['true_index']].capitalize()

        max_score = prediction['predictors_scores'].max().sigmoid()
        fig.suptitle(f'Prediction: {pred_label} ({max_score * 100:.2f}%). Ground Truth: {true_label}', fontsize=24, fontweight='bold', y=1.1)

        # Save figure
        file_path = os.path.splitext(os.path.basename(instance['segmentations'].input_image_path))[0]
        out_path = os.path.join(args.output_dir, f'{file_path}.jpg')
        fig.savefig(out_path, bbox_inches='tight') # Recompute fig dims when saving because of raised suptitle
        plt.close(fig)

    # %% Compute and save stats
    all_total_scores = [c['total_scores'] for c in all_contributions]
    avg_contributions = {
        k : torch.tensor([d[k] for d in all_total_scores]).abs().mean().item()
        for k in all_total_scores[0].keys()
    }

    stats = {}
    stats['avg_contribution_mass'] = avg_contributions
    stats['accuracy'] = accuracy.compute().item()

    with open(os.path.join(args.output_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    # %%
