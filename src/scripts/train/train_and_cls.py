# %%
import os # Change DesCo CUDA device here
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Prepend to path so starts searching at src first
import sys
sys.path = [os.path.join(os.path.dirname(__file__), '..')] + sys.path

from llm import LLMClient
from kb_ops import kb_from_img_dir, add_global_negatives
from kb_ops.build_kb import label_from_path, label_from_directory
from model.concept import ConceptKBConfig
from kb_ops.train_test_split import split
from kb_ops.dataset import FeatureDataset, extend_with_global_negatives
from kb_ops import ConceptKBFeatureCacher, ConceptKBFeaturePipeline, ConceptKBExampleSampler, ConceptKBFeaturePipelineConfig
from image_processing import LocalizerAndSegmenterConfig
from model.concept import ConceptKB
import logging, coloredlogs
from feature_extraction.trained_attrs import N_ATTRS_DINO
from kb_ops.train import ConceptKBTrainer
import wandb
import jsonargparse as argparse
from itertools import chain
from scripts.utils import set_feature_paths, get_timestr

logger = logging.getLogger(__name__)

def parse_args(parser: argparse.ArgumentParser, cl_args: list[str] = None, config_str: str = None) -> argparse.Namespace:
    if config_str:
        args = parser.parse_string(config_str)
    else:
        args = parser.parse_args(cl_args)

    return args

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action=argparse.ActionConfigFile)

    parser.add_argument('--img_dir', type=str,
                        default='/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/xiaomeng_augmented_data_v3',
                        help='Path to directory of images or preprocessed segmentations')

    parser.add_argument('--extract_label_from', choices=['path', 'directory'], default='path',
                        help='Whether to extract concept labels from image paths or containing directories')

    parser.add_argument('--negatives_img_dir', type=str, default='/shared/nas2/blume5/fa23/ecole/data/imagenet/negatives_rand_1k',
                        help='Path to directory of negative example images')

    parser.add_argument('--ckpt_path', help='If provided, loads the ConceptKB from this pickle file instead of creating a new one.')

    # Cache
    parser.add_argument('--cache.root', default='/shared/nas2/blume5/fa23/ecole/cache/xiaomeng_augmented_data_v3', help='Directory to save example cache')
    parser.add_argument('--cache.segmentations', default='segmentations', help='Subdirectory of cache_dir to save segmentations')
    parser.add_argument('--cache.features', default='features', help='Subdirectory of cache_dir to save segmentations')

    parser.add_argument('--cache.negatives.root', default='/shared/nas2/blume5/fa23/ecole/cache/imagenet_rand_1k', help='Directory to save negative example feature cache')
    parser.add_argument('--cache.negatives.features', default='features', help='Subdirectory of cache_dir to save negative example features')
    parser.add_argument('--cache.negatives.segmentations', default='segmentations', help='Subdirectory of cache_dir to save negative example segmentations')

    parser.add_argument('--wandb_project', type=str, default='ecole_mo9_demo', help='WandB project name')
    parser.add_argument('--wandb_dir', default='/shared/nas2/blume5/fa23/ecole', help='WandB log directory')

    # Feature pipeline
    parser.add_argument('--feature_pipeline_config', type=ConceptKBFeaturePipelineConfig, default=ConceptKBFeaturePipelineConfig()),
    parser.add_argument('--loc_and_seg_config', type=LocalizerAndSegmenterConfig, default=LocalizerAndSegmenterConfig())

    # Predictor
    parser.add_argument('--predictor.use_ln', type=bool, default=False, help='Whether to use LayerNorm')
    parser.add_argument('--predictor.use_probabilities', type=bool, default=False, help='Whether to sigmoid raw scores instead of layer-norming them for prediction')
    parser.add_argument('--predictor.use_full_img', type=bool, default=True, help='Whether to use full image as input')
    parser.add_argument('--predictor.use_regions', type=bool, default=True, help='Whether to use regions as input')
    parser.add_argument('--predictor.encode_class_in_zs_attr', type=bool, default=False, help='Whether to encode class in zero-shot attributes')

    # Training
    parser.add_argument('--train.n_epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--train.lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train.backward_every_n_concepts', type=int, default=10, help='Number of concepts to add losses for between backward calls. Higher values are faster but consume more memory')
    parser.add_argument('--train.imgs_per_optim_step', type=int, default=4, help='Number of images to accumulate gradients over before stepping optimizer')
    parser.add_argument('--train.ckpt_every_n_epochs', type=int, default=1, help='Number of epochs between checkpoints')
    parser.add_argument('--train.ckpt_dir', type=str, default='/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb', help='Directory to save model checkpoints')

    parser.add_argument('--train.use_concepts_as_negatives', type=bool, default=False, help='Whether to use other concepts as negatives')
    parser.add_argument('--train.use_global_negatives', type=bool, default=True, help='Whether to use global negative examples during training')
    parser.add_argument('--train.limit_global_negatives', type=int, help='The number of global negative examples to use during training. If None, uses all')

    parser.add_argument('--train.split', type=tuple[float,float,float], default=(.6, .2, .2), help='Train, val, test split ratios')

    parser.add_argument('--train.use_descandants_as_positives', type=bool, default=True, help='Whether to train more general concepts (hypernyms) with its more specific instances (hyponyms)')
    parser.add_argument('--train.negatives_strategy', choices=['use_all_concepts_as_negatives', 'use_siblings_as_negatives', 'only_positives'],
                        default='use_siblings_as_negatives')

    return parser

def main(args: argparse.Namespace, parser: argparse.ArgumentParser, concept_kb: ConceptKB = None):
    # %%
    run = wandb.init(project=args.wandb_project, config=args.as_flat(), dir=args.wandb_dir, reinit=True)
    # run = None # Comment me to use wandb

    # Import here so DesCo sees the CUDA device change
    from feature_extraction import (
        build_feature_extractor,
        build_desco,
        build_sam
    )
    from image_processing import build_localizer_and_segmenter

    # %%
    # loc_and_seg = build_localizer_and_segmenter(build_sam(), build_desco())
    loc_and_seg = build_localizer_and_segmenter(
        build_sam(),
        None,
        config=LocalizerAndSegmenterConfig(**args.loc_and_seg_config.as_dict())
    ) # Don't load DesCo to save startup time
    feature_extractor = build_feature_extractor()
    feature_pipeline = ConceptKBFeaturePipeline(
        loc_and_seg,
        feature_extractor,
        config=ConceptKBFeaturePipelineConfig(**args.feature_pipeline_config.as_dict())
    )

    # %% Initialize concept KB
    if args.ckpt_path:
        assert concept_kb is None, 'Cannot load a ConceptKB from a checkpoint and also pass one in as an argument'
        concept_kb = ConceptKB.load(args.ckpt_path)

    else:
        label_extractor = label_from_path if args.extract_label_from == 'path' else label_from_directory
        concept_kb = kb_from_img_dir(args.img_dir, label_from_path_fn=label_extractor) if concept_kb is None else concept_kb

        if args.train.use_global_negatives:
            add_global_negatives(concept_kb, args.negatives_img_dir, limit=args.train.limit_global_negatives)

        concept_kb.initialize(ConceptKBConfig(
            encode_class_in_zs_attr=args.predictor.encode_class_in_zs_attr,
            n_trained_attrs=N_ATTRS_DINO,
            use_ln=args.predictor.use_ln,
            use_probabilities=args.predictor.use_probabilities,
            use_full_img=args.predictor.use_full_img,
            use_regions=args.predictor.use_regions
        ), llm_client=LLMClient())
        # )) # Uncomment me and comment above to test with no ZS attributes to avoid paying Altman

    # %% Train concept detectors
    trainer = ConceptKBTrainer(concept_kb, feature_pipeline, wandb_run=run)

    # %% Set cached segmentation, feature paths if they are provided and exist
    # NOTE Must recompute features every time we train a new model, as the number of zero-shot attributes
    # is nondeterministic from the LLM
    # features_dir = os.path.join(args.cache.root, args.cache.features)
    segmentations_dir = os.path.join(args.cache.root, args.cache.segmentations)
    set_feature_paths(concept_kb, segmentations_dir=segmentations_dir)

    if args.train.use_global_negatives:
        neg_segmentations_dir = os.path.join(args.cache.negatives.root, args.cache.negatives.segmentations)
        set_feature_paths(concept_kb.global_negatives, segmentations_dir=neg_segmentations_dir)

    # Prepare examples
    sampler = ConceptKBExampleSampler(concept_kb)

    all_examples = list(chain.from_iterable(c.examples for c in concept_kb))
    all_labels = list(chain.from_iterable([[c.name for _ in c.examples] for c in concept_kb])) # No way currently to indicate concept-specific negatives from files

    (train_exs, train_labels), (val_exs, val_labels), (test_exs, test_labels) = split(all_examples, all_labels, split=args.train.split)
    train_concepts_to_train_per_example = sampler.get_concepts_to_train_per_example(
        train_exs,
        use_descendants_as_positives=args.train.use_descandants_as_positives,
        negatives_strategy=args.train.negatives_strategy
    )

    val_concepts_to_train_per_example = sampler.get_concepts_to_train_per_example(
        val_exs,
        use_descendants_as_positives=args.train.use_descandants_as_positives,
        negatives_strategy=args.train.negatives_strategy
    )

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

    # Create datasets
    train_paths = [ex.image_features_path for ex in train_exs]
    val_paths = [ex.image_features_path for ex in val_exs]

    train_ds = FeatureDataset(train_paths, train_labels, concepts_to_train_per_example=train_concepts_to_train_per_example)
    val_ds = FeatureDataset(val_paths, val_labels, concepts_to_train_per_example=val_concepts_to_train_per_example)

    # Consider splitting global negatives into train and val sets?
    extend_with_global_negatives(train_ds, concept_kb.global_negatives)

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

# %%
if __name__ == '__main__':
    coloredlogs.install(level=logging.INFO)
    parser = get_parser()
    args = parse_args(parser)
    main(args, parser)