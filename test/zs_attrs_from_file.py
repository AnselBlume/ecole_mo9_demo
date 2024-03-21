# %%
import os # Change DesCo CUDA device here
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from llm import LLMClient
from kb_ops import kb_from_img_dir
from model.concept import ConceptKBConfig
from kb_ops.train_test_split import split_from_directory, split_from_paths
from kb_ops.dataset import FeatureDataset
from kb_ops import ConceptKBFeatureCacher, ConceptKBFeaturePipeline
from typing import Any
import logging, coloredlogs
from feature_extraction.trained_attrs import N_ATTRS_DINO
from kb_ops.train import ConceptKBTrainer
import wandb
import jsonargparse as argparse
from itertools import chain
from .utils import set_feature_paths, get_timestr

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action=argparse.ActionConfigFile)

    parser.add_argument('--img_dir', type=str,
                        default='/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/xiaomeng_augmented_data_v1',
                        help='Path to directory of images or preprocessed segmentations')

    parser.add_argument('--predictor.use_ln', type=bool, default=True, help='Whether to use LayerNorm')
    parser.add_argument('--predictor.use_probabilities', type=bool, default=False, help='Whether to sigmoid raw scores instead of layer-norming them for prediction')
    parser.add_argument('--predictor.use_full_img', type=bool, default=True, help='Whether to use full image as input')
    parser.add_argument('--predictor.use_regions', type=bool, default=True, help='Whether to use regions as input')
    parser.add_argument('--predictor.encode_class_in_zs_attr', type=bool, default=False, help='Whether to encode class in zero-shot attributes')

    return parser

# %%
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    # args = parser.parse_args([]) # For debugging or jupyter notebook

    # %% Initialize concept KB
    concept_kb = kb_from_img_dir(args.img_dir)

    # TODO load concept_to_zs_attrs dict. This should map from concept names to
    # dictionaries of the format processed in ConceptKB.init_zs_attrs
    concept_to_zs_attrs: dict[str,Any] = None

    # %%
    concept_kb.initialize(ConceptKBConfig(
        encode_class_in_zs_attr=args.predictor.encode_class_in_zs_attr,
        img_feature_dim=768, # Set arbitrarily for this testing script
        n_trained_attrs=N_ATTRS_DINO,
        use_ln=args.predictor.use_ln,
        use_probabilities=args.predictor.use_probabilities,
        use_full_img=args.predictor.use_full_img,
        use_regions=args.predictor.use_regions
    ), llm_client=LLMClient(), concept_to_zs_attrs = None) # TODO load me

    # %% Check the zero-shot attributes
    for concept in concept_kb:
        print(f'Concept: {concept}, zs_attrs: {concept.zs_attributes}')
        print('')