# %%
import os # TODO Change DesCo CUDA device here
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from kb_ops import kb_from_img_dir
from model.concept import ConceptKBConfig
from controller import Controller
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from kb_ops.train_test_split import split_from_directory
from vis_utils import image_from_masks
from kb_ops.dataset import ImageDataset, PresegmentedDataset
import logging, coloredlogs
from feature_extraction.trained_attrs import N_ATTRS_SUBSET
from kb_ops.train import ConceptKBTrainer
import wandb
from datetime import datetime
import jsonargparse as argparse

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

    parser.add_argument('--presegmented_dir',
                        default='/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/xiaomeng_augmented_data_segmentations',
                        help='Path to directory of preprocessed segmentations')

    parser.add_argument('--wandb_project', type=str, default='ecole_mo9_demo', help='WandB project name')
    parser.add_argument('--wandb_dir', default='/shared/nas2/blume5/fa23/ecole', help='WandB log directory')

    parser.add_argument('--predictor.use_ln', type=bool, default=True, help='Whether to use LayerNorm')
    parser.add_argument('--predictor.use_full_img', type=bool, default=True, help='Whether to use full image as input')
    parser.add_argument('--predictor.encode_class_in_zs_attr', type=bool, default=False, help='Whether to encode class in zero-shot attributes')

    parser.add_argument('--train.n_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--train.lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train.backward_every_n_concepts', type=int, default=None, help='Number of concepts to accumulate gradients over')
    parser.add_argument('--train.imgs_per_optim_step', type=int, default=4, help='Number of images to accumulate gradients over before stepping optimizer')
    parser.add_argument('--train.ckpt_every_n_epochs', type=int, default=1, help='Number of epochs to save model')
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

    # %% Split images into train, val, test
    (trn_p, trn_l), (val_p, val_l), (tst_p, tst_l) = split_from_directory(args.img_dir)

    # Build controller for segmentation
    controller = Controller(
        build_sam(),
        build_desco(),
        concept_kb
    )

    # %%
    feature_extractor = build_feature_extractor()

    # %%
    concept_kb.initialize(ConceptKBConfig(
        encode_class_in_zs_attr=args.predictor.encode_class_in_zs_attr,
        img_feature_dim=feature_extractor.clip.config.projection_dim,
        n_trained_attrs=N_ATTRS_SUBSET,
        use_ln=args.predictor.use_ln,
        use_full_img=args.predictor.use_full_img,
    ), llm_client=controller.llm_client)
    # )) # Uncomment me and comment above to test with no ZS attributes to avoid paying Altman

    # %% Train concept detectors
    trainer = ConceptKBTrainer(concept_kb, feature_extractor, controller, run)

    # %%
    if args.presegmented_dir:
        trn_p = [os.path.join(args.presegmented_dir, os.path.basename(os.path.splitext(p)[0]) + '.pkl') for p in trn_p]
        val_p = [os.path.join(args.presegmented_dir, os.path.basename(os.path.splitext(p)[0]) + '.pkl') for p in val_p]

        train_ds = PresegmentedDataset(trn_p, trn_l)
        val_ds = PresegmentedDataset(val_p, val_l)

    else:
        train_ds = ImageDataset(trn_p, trn_l)
        val_ds = ImageDataset(val_p, val_l)

    concept_kb.to('cuda')

    # Save arguments as yaml
    checkpoint_dir = os.path.join(args.train.ckpt_dir, get_timestr())
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
