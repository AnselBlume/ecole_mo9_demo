# %%
import os # TODO Change DesCo CUDA device here
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from llm import LLMClient
from kb_ops import kb_from_img_dir
from model.concept import ConceptKBConfig
from kb_ops.train_test_split import split_from_directory, split_from_paths
from kb_ops.dataset import FeatureDataset
from kb_ops import ConceptKBFeatureCacher, ConceptKBFeaturePipeline
import logging, coloredlogs
from feature_extraction.trained_attrs import N_ATTRS_SUBSET
from kb_ops.train import ConceptKBTrainer
import wandb
from datetime import datetime
import jsonargparse as argparse
from itertools import chain

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

def get_timestr():
    return datetime.now().strftime('%Y_%m_%d-%H:%M:%S')

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action=argparse.ActionConfigFile)

    parser.add_argument('--img_dir', type=str,
                        default='/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/xiaomeng_augmented_data',
                        help='Path to directory of images or preprocessed segmentations')

    parser.add_argument('--cache.root', default='/shared/nas2/blume5/fa23/ecole/cache/xiaomeng_augmented_data', help='Directory to save feature cache')
    parser.add_argument('--cache.segmentations', default='segmentations', help='Subdirectory of cache_dir to save segmentations')
    parser.add_argument('--cache.features', default='features', help='Subdirectory of cache_dir to save segmentations')

    parser.add_argument('--wandb_project', type=str, default='ecole_mo9_demo', help='WandB project name')
    parser.add_argument('--wandb_dir', default='/shared/nas2/blume5/fa23/ecole', help='WandB log directory')

    parser.add_argument('--predictor.use_ln', type=bool, default=True, help='Whether to use LayerNorm')
    parser.add_argument('--predictor.use_full_img', type=bool, default=True, help='Whether to use full image as input')
    parser.add_argument('--predictor.use_regions', type=bool, default=True, help='Whether to use regions as input')
    parser.add_argument('--predictor.encode_class_in_zs_attr', type=bool, default=False, help='Whether to encode class in zero-shot attributes')

    parser.add_argument('--train.n_epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--train.lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train.backward_every_n_concepts', type=int, default=10, help='Number of concepts to add losses for between backward calls. Higher values are faster but consume more memory')
    parser.add_argument('--train.imgs_per_optim_step', type=int, default=4, help='Number of images to accumulate gradients over before stepping optimizer')
    parser.add_argument('--train.ckpt_every_n_epochs', type=int, default=1, help='Number of epochs between checkpoints')
    parser.add_argument('--train.ckpt_dir', type=str, default='/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb', help='Directory to save model checkpoints')

    return parser

# %%
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    # args = parser.parse_args([]) # For debugging or jupyter notebook

    # %%
    run = wandb.init(project='ecole_mo9_demo', config=args.as_flat(), dir=args.wandb_dir, reinit=True)
    # run = None # Comment me to use wandb

    # %% Initialize concept KB
    concept_kb = kb_from_img_dir(args.img_dir)

    # Import here so DesCo sees the CUDA device change
    from feature_extraction import (
        build_feature_extractor,
        build_desco,
        build_sam,
    )
    from image_processing import build_localizer_and_segmenter

    # %%
    # loc_and_seg = build_localizer_and_segmenter(build_sam(), build_desco())
    loc_and_seg = build_localizer_and_segmenter(build_sam(), None) # TODO Comment me and use above when not testing
    feature_extractor = build_feature_extractor()
    feature_pipeline = ConceptKBFeaturePipeline(concept_kb, loc_and_seg, feature_extractor)

    # %%
    concept_kb.initialize(ConceptKBConfig(
        encode_class_in_zs_attr=args.predictor.encode_class_in_zs_attr,
        img_feature_dim=feature_extractor.clip.config.projection_dim,
        n_trained_attrs=N_ATTRS_SUBSET,
        use_ln=args.predictor.use_ln,
        use_full_img=args.predictor.use_full_img,
        use_regions=args.predictor.use_regions
    ), llm_client=LLMClient())
    # )) # Uncomment me and comment above to test with no ZS attributes to avoid paying Altman

    # %% Train concept detectors
    trainer = ConceptKBTrainer(concept_kb, feature_pipeline, run)

    # %% Set cached segmentation, feature paths if they are provided and exist
    features_dir = os.path.join(args.cache.root, args.cache.features)
    if os.path.exists(features_dir):
        # Store pre-computed feature paths in concept examples
        for concept in concept_kb:
            for example in concept.examples:
                basename = os.path.basename(os.path.splitext(example.image_path)[0]) + '.pkl'
                example.image_features_path = os.path.join(features_dir, basename)

    segmentations_dir = os.path.join(args.cache.root, args.cache.segmentations)
    if os.path.exists(segmentations_dir):
        # Store presegmented paths in concept examples
        for concept in concept_kb:
            for example in concept.examples:
                basename = os.path.basename(os.path.splitext(example.image_path)[0]) + '.pkl'
                example.image_segmentations_path = os.path.join(segmentations_dir, basename)

    # Pre-cache features
    cacher = ConceptKBFeatureCacher(
        concept_kb=concept_kb,
        feature_pipeline=feature_pipeline,
        cache_dir=args.cache.root,
        segmentations_sub_dir=args.cache.segmentations,
        features_sub_dir=args.cache.features
    )
    cacher.cache_segmentations()
    cacher.cache_features()

    # Collect all segmentation paths we just added
    all_feature_paths = list(chain.from_iterable([
        [ex.image_features_path for ex in c.examples]
        for c in concept_kb
    ]))


    (trn_p, trn_l), (val_p, val_l), (tst_p, tst_l) = split_from_paths(all_feature_paths)
    train_ds = FeatureDataset(trn_p, trn_l)
    val_ds = FeatureDataset(val_p, val_l)

    concept_kb.to('cuda')

    # Save arguments as yaml
    run_id = f'{get_timestr()}-{run.id}' if run else get_timestr()
    checkpoint_dir = os.path.join(args.train.ckpt_dir, run_id)

    os.makedirs(checkpoint_dir)

    parser.save(args, os.path.join(checkpoint_dir, 'args.yaml'))

    # Train
    trainer.train(
        train_ds=train_ds,
        val_ds=val_ds,
        n_epochs=args.train.n_epochs,
        lr=args.train.lr,
        backward_every_n_concepts=args.train.backward_every_n_concepts,
        imgs_per_optim_step=args.train.imgs_per_optim_step,
        ckpt_every_n_epochs=args.train.ckpt_every_n_epochs,
        ckpt_dir=checkpoint_dir
    )

# %%
