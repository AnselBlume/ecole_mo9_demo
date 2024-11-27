# %%
import os # Change DesCo CUDA device here
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
import yaml
from tqdm import tqdm
from model.concept import ConceptKB, ConceptExample
from kb_ops.train_test_split import split
from kb_ops.dataset import list_collate, FeatureDataset
from kb_ops import ConceptKBFeaturePipeline
from kb_ops.build_kb import list_paths, label_from_directory, label_from_path
import logging, coloredlogs
from feature_extraction.trained_attrs import N_ATTRS_DINO
from kb_ops import ConceptKBTrainer, ConceptKBPredictor
from scripts.utils import parse_args, set_feature_paths
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
import jsonargparse
from utils import open_image
from itertools import chain

logger = logging.getLogger(__name__)

def get_parser():
    parser = jsonargparse.ArgumentParser()

    parser.add_argument('--ckpt_path', help='Path to model checkpoint to evaluate')
    parser.add_argument('--output_dir', help='Directory to save predictions to', default='predictions')

    parser.add_argument('--predict.img_dir', type=str, help='Directory of images to output predictions for. Subfolders should indicate label names.')
    parser.add_argument('--predict.segmentations_dir', type=str, help='Directory of segmentations to output predictions for. Subfolders should indicate label names.')
    parser.add_argument('--predict.features_dir', type=str, help='Directory of features to output predictions for. Subfolders should indicate label names.')

    parser.add_argument('--predict.test_images', action='store_true', help='Predict the ConceptKB\'s images with no region or image annotations')
    parser.add_argument('--predict.test_ground_truth_regions', action='store_true', help='Predict the ConceptKB\'s ground truth regions in the test set')

    parser.add_argument('--predict.function', choices=['hierarchical', 'flat'], default='hierarchical', help='Prediction function to use')

    parser.add_argument('--unk_threshold', type=float, default=.6, help='Confidence threshold below which unknown will be predicted')

    parser.add_argument('--cache.root', default='/shared/nas2/blume5/fa23/ecole/cache/predictions_temp', help='Directory to save feature cache')
    parser.add_argument('--cache.segmentations', default='segmentations', help='Subdirectory of cache_dir to save segmentations')
    parser.add_argument('--cache.features', default='features', help='Subdirectory of cache_dir to save segmentations')

    return parser

def predict_from_features():
    pass

def predict_images(image_paths: list[str]):
    pass

def _select_predict_fn(predictor: ConceptKBPredictor, args: jsonargparse.Namespace):
    if args.predict.function == 'hierarchical':
        return predictor.hierarchical_predict
    elif args.predict.function == 'flat':
        return predictor.predict
    else:
        raise ValueError(f'Invalid prediction function {args.predict.function}')

def _build_predictor(concept_kb: ConceptKB, ckpt_config: dict) -> tuple[ConceptKBPredictor, ConceptKBTrainer]:
    # Import here so DesCo sees the CUDA device change
    from feature_extraction import build_feature_extractor, build_sam
    from image_processing import build_localizer_and_segmenter

    loc_and_seg = build_localizer_and_segmenter(build_sam(), None)
    feature_extractor = build_feature_extractor()
    feature_pipeline = ConceptKBFeaturePipeline(loc_and_seg, feature_extractor)

    # Set some key attributes to match what the checkpoint was trained with
    feature_pipeline.config.use_zs_attr_scores = ckpt_config.get('feature_pipeline_config', {}) \
                                                            .get('use_zs_attr_scores', True)

    feature_pipeline.loc_and_seg.config.do_localize = ckpt_config.get('loc_and_seg_config', {}) \
                                                                 .get('do_localize', True)

    feature_pipeline.loc_and_seg.config.do_segment = ckpt_config.get('loc_and_seg_config', {}) \
                                                                .get('do_segment', False)

    return ConceptKBPredictor(concept_kb, feature_pipeline), ConceptKBTrainer(concept_kb, feature_pipeline)

def _load_checkpoint_config(ckpt_path: str):
    ckpt_dir = os.path.dirname(ckpt_path)
    with open(os.path.join(ckpt_dir, 'args.yaml'), 'r') as f:
        return yaml.safe_load(f)

def _load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)

def main(args, parser: jsonargparse.ArgumentParser):
    os.makedirs(args.output_dir, exist_ok=True)
    parser.save(args, os.path.join(args.output_dir, 'args.yaml'), overwrite=True)

    concept_kb = ConceptKB.load(args.ckpt_path)
    concept_kb.cuda().eval()

    ckpt_config = _load_checkpoint_config(args.ckpt_path)
    predictor, trainer = _build_predictor(concept_kb, ckpt_config)

    # Validate arguments
    if sum([
        bool(args.predict.test_images),
        bool(args.predict.test_ground_truth_regions),
        bool(args.predict.img_dir),
        bool(args.predict.segmentations_dir),
        bool(args.predict.features_dir)
    ]) != 1:
        raise ValueError(
            'Exactly one of --predict.test_images, --predict.test_ground_truth_regions, '
            + '--predict.img_dir, --predict.segmentations_dir, or --predict.features_dir must be set'
        )

    # Automatically select the test set
    paths = None
    load_fn = None

    if args.predict.test_images or args.predict.test_ground_truth_regions:
        load_fn = _load_pickle

        # Split examples in the same way they were at train time
        all_examples: list[ConceptExample] = list(chain.from_iterable(c.examples for c in concept_kb))
        all_labels: list[str] = list(chain.from_iterable([[c.name for _ in c.examples] for c in concept_kb])) # No way currently to indicate concept-specific negatives from files

        (train_exs, train_labels), (val_exs, val_labels), (test_exs, test_labels) = split(all_examples, all_labels, split=ckpt_config['train']['split'])

        if args.predict.test_images:
            # Gather features for images with no region annotations (including for parts)
            # all_images_and_features = [(ex.image_path, ex.image_features_path) for ex in test_exs if not ex.is_negative]

            # Don't include images with regions as they may have been used in training since an example
            # is an image-region pair
            images_with_regions = set()
            for ex in test_exs:
                if ex.object_mask_rle_json_path:
                    images_with_regions.add(ex.image_path)

            paths = []
            labels = []
            positive_test_exs = [ex for ex in test_exs if not ex.is_negative]
            for example in positive_test_exs:
                if example.image_path not in images_with_regions:
                    paths.append(example.image_features_path)
                    labels.append(example.concept_name)

        elif args.predict.test_ground_truth_regions:
            # TODO exclude components
            # Assume we have ground truth regions we want to classify
            paths = [ex.image_features_path for ex in test_exs]
            labels = test_labels

    # Test set provided (or set via automatic selection)
    elif args.predict.img_dir:
        load_fn = open_image
        paths = list_paths(args.predict.img_dir, exts=['.jpg', '.png', '.webp', '.jpeg'])
        labels = [label_from_directory(path) for path in paths]

    elif args.predict.segmentations_dir or args.predict.features_dir:
        load_fn = _load_pickle
        paths = list_paths(args.predict.img_dir, exts=['.pkl'])
        labels = [label_from_directory(path) for path in paths]

    else:
        raise RuntimeError('Invalid argument combination')

    outputs = []
    predict_fn = _select_predict_fn(predictor, args)
    prog_bar = tqdm(zip(paths, labels), total=len(paths))
    for path, label in prog_bar: # list_paths already sorts the paths
        data = load_fn(path)
        prediction = predict_fn(image_data=data, unk_threshold=args.unk_threshold)
        output = {
            'label': label,
            'path': path,
            'prediction': prediction
        }
        outputs.append(output)

    predictions_path = os.path.join(args.output_dir, 'predictions.pkl')
    with open(predictions_path, 'wb') as f:
        pickle.dump(outputs, f)

    # Validation evaluation
    # dataset = FeatureDataset(paths, labels)
    # dl = DataLoader(dataset, collate_fn=list_collate)
    # print(trainer.validate(dl))

    logger.info('Done')

# %%
if __name__ == '__main__':
    coloredlogs.install(level=logging.INFO)

    parser = get_parser()
    args = parse_args(parser, config_str='''
        ckpt_path: /shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_11_15-03:04:28-f7fwkjz3/concept_kb_epoch_13.pt
        predict:
            # test_ground_truth_regions: true
            test_images: true

        # output_dir: /shared/nas2/blume5/fa23/ecole/test_gt_region_predictions
        output_dir: /shared/nas2/blume5/fa23/ecole/test_images_predictions
    ''')

    main(args, parser)

# %%
