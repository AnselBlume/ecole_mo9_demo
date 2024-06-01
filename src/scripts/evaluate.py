# %%
import os # Change DesCo CUDA device here
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.concept import ConceptKB
from kb_ops.train_test_split import split_from_directory, split_from_paths
from kb_ops.dataset import PresegmentedDataset, list_collate
from kb_ops import ConceptKBFeaturePipeline
import logging, coloredlogs
from feature_extraction.trained_attrs import N_ATTRS_DINO
from kb_ops.train import ConceptKBTrainer
from .utils import set_feature_paths
from torchmetrics import Accuracy
import jsonargparse as argparse
from itertools import chain
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action=argparse.ActionConfigFile)

    parser.add_argument('--ckpt_path', help='Path to model checkpoint to evaluate')

    parser.add_argument('--img_dir', type=str,
                        default='/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/xiaomeng_augmented_data',
                        help='Path to directory of images or preprocessed segmentations')

    parser.add_argument('--cache.root', default='/shared/nas2/blume5/fa23/ecole/cache/xiaomeng_augmented_data', help='Directory to save feature cache')
    parser.add_argument('--cache.segmentations', default='segmentations', help='Subdirectory of cache_dir to save segmentations')
    parser.add_argument('--cache.features', default='features', help='Subdirectory of cache_dir to save segmentations')

    return parser

# %%
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # Load concept KB
    concept_kb = ConceptKB.load(args.ckpt_path)

    # Import here so DesCo sees the CUDA device change
    from feature_extraction import (
        build_feature_extractor,
        build_desco,
        build_sam,
    )
    from image_processing import build_localizer_and_segmenter

    loc_and_seg = build_localizer_and_segmenter(build_sam(), build_desco())
    feature_extractor = build_feature_extractor()
    feature_pipeline = ConceptKBFeaturePipeline(concept_kb, loc_and_seg, feature_extractor)

    # Build Trainer
    trainer = ConceptKBTrainer(concept_kb, feature_pipeline)

    # NOTE We won't cache the features here, since we're doing one forward pass per image anyways. We also won't
    # attempt to use cached features to be safe, as if the features have been regenerated since the checkpoint to be
    # evaluated, the features will not match the model due to different numbers of zero-shot attributes due to LLM
    # nondeterminism
    all_segmentation_paths = list(chain.from_iterable([
        [ex.image_segmentations_path for ex in c.examples] # These are saved in the checkpoint, so no need to set
        for c in concept_kb
    ]))

    (trn_p, trn_l), (val_p, val_l), (tst_p, tst_l) = split_from_paths(all_segmentation_paths)
    test_ds = PresegmentedDataset(tst_p, tst_l)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=list_collate)

    concept_kb.to('cuda')

    results = trainer.validate(test_dl)
    logger.info(f'Component Accuracy: {results.component_accuracy}')
    logger.info(f'Non-Component Accuracy: {results.non_component_accuracy}')

# %%
