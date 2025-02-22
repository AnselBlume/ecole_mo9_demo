# %%
import os # Change DesCo CUDA device here
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Prepend to path so starts searching at src first
import sys
sys.path = [os.path.join(os.path.dirname(__file__), '../..')] + sys.path

from kb_ops.build_kb import label_from_path, label_from_directory
from kb_ops import kb_from_img_dir, kb_from_img_and_mask_dirs
import logging, coloredlogs
from scripts.train.train_and_cls import main, parse_args, get_parser as get_base_parser
from scripts.train.parse_hierarchy import ConceptGraphParser

logger = logging.getLogger(__name__)

def get_parser():
    parser = get_base_parser()
    parser.add_argument('--hierarchy_config', type=dict, help='The hierarchy config as a yaml string')
    parser.add_argument('--hierarchy_config_path', type=str, help='Path to hierarchy config yaml')

    return parser

if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG)

    parser = get_parser()
    args = parse_args(parser, config_str='''
        # ckpt_path: /shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_06_01-00:57:58-85pf2vzt-no_bp_no_cj_no_localize/concept_kb_epoch_50.pt

        # img_dir: /shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_and_guns_v4

        # Image directory on both network attached storage and hard disk
        img_dir: /shared/nas2/blume5/fa24/concept_downloading/data/image_annotations/24-10-26/annotations/merged_annotations/images
        # img_dir: /scratch/blume5/merged_annotations/images

        # Object mask directory on both network attached storage and hard disk
        object_mask_rle_dir: /shared/nas2/blume5/fa24/concept_downloading/data/image_annotations/24-10-26/annotations/merged_annotations/masks
        # object_mask_rle_dir: /scratch/blume5/merged_annotations/masks

        train:
            # limit_global_negatives: 5
            split: [.8, 0, .2]
            n_epochs: 50
            lr: 1e-3
            dataloader_kwargs:
                num_workers: 0
                pin_memory: false
                persistent_workers: false

            do_minimal: false
            in_memory: true

        extract_label_from: directory

        cache:
            # XXX This MUST be changed to a directory which we don't care about to not overwrite checkpoints
            # root: /shared/nas2/blume5/fa23/ecole/cache/airplanes_and_guns_v4/all_v1_localize_use_containing_concepts
            root: /shared/nas2/blume5/fa23/ecole/cache/2024_december_1k/v1_no_zs_attrs
            # root: /scratch/blume5/cache/2024_december_1k/v1

            # Comment this to use standard negatives cache with SAM region segmentations
            negatives:
                # root: /shared/nas2/blume5/fa23/ecole/cache/imagenet_rand_1k_no_sam_segmentations
                root: /shared/nas2/blume5/fa23/ecole/cache/2024_december_1k/v1

            infer_localize_from_component: false

        wandb_project: ecole_december_2024

        feature_pipeline_config:
            # Change this to true in order to run component detection with DesCo
            compute_component_concept_scores: false

            use_zs_attr_scores: false

        loc_and_seg_config:
            do_localize: true
            do_segment: false

        example_sampler_config:
            use_descendants_as_positives: true
            use_containing_concepts_for_positives: false

        # hierarchy_config_path: /shared/nas2/blume5/fa24/concept_downloading/data/image_annotations/24-10-22/annotations/merged_annotations/graph_small.yaml
        hierarchy_config_path: /shared/nas2/blume5/fa24/concept_downloading/data/image_annotations/24-10-26/annotations/merged_annotations/graph.yaml
        # hierarchy_config_path: /scratch/blume5/merged_annotations/graph.yaml

        # hierarchy_config:
        #     concepts:
        #         - airplane
        #         - transport plane
        #         - cargo jet
        #         - passenger plane
        #         - biplane
        #         - fighter jet

        #         - gun
        #         - sniper rifle
        #         - machine gun
        #         - pistol
        #         # - assault rifle

        #         # Component concepts
        #         - propulsion component
        #         - wings
        #         - wing-mounted engine
        #         - bulky fuselage
        #         - openable nose
        #         - row of windows
        #         - double wings
        #         - fixed landing gear
        #         - propeller
        #         - afterburner

        #         - trigger
        #         - barrel
        #         - grip
        #         - bipod or tripod
        #         - scope
        #         - grip with magazine
        #         - ammunition belt
        #         # - detachable box magazine

        #     instance_graph:
        #         airplane: ['transport plane', 'biplane', 'fighter jet']
        #         transport plane: ['cargo jet', 'passenger plane']

        #         propulsion component: ['wing-mounted engine', 'afterburner', 'propeller']
        #         wings: ['double wings']

        #         gun: ['assault rifle', 'sniper rifle', 'machine gun', 'pistol']
        #         grip: ['grip with magazine']

        #     component_graph:
        #         airplane: ['wings', 'propulsion component']
        #         transport plane: ['wing-mounted engine']
        #         cargo jet: ['bulky fuselage', 'openable nose']
        #         passenger plane: ['row of windows']
        #         biplane: ['double wings', 'fixed landing gear', 'propeller']
        #         fighter jet: ['afterburner']

        #         gun: ['trigger', 'barrel', 'grip']
        #         pistol: ['grip with magazine']
        #         sniper rifle: ['scope', 'bipod or tripod']
        #         machine gun: ['bipod or tripod', 'ammunition belt']
        ''')

    if args.ckpt_path:
        concept_kb = None # Will be loaded in main

    else:
        # Construct KB from images
        label_extractor = label_from_path if args.extract_label_from == 'path' else label_from_directory

        if args.object_mask_rle_dir:
            concept_kb = kb_from_img_and_mask_dirs(args.img_dir, args.object_mask_rle_dir, label_from_path_fn=label_extractor)
        else: # No masks
            concept_kb = kb_from_img_dir(args.img_dir, label_from_path_fn=label_extractor)

        # Apply graph connectivity and create any concepts without images
        graph = ConceptGraphParser.parse_graph(path=args.hierarchy_config_path, config=args.hierarchy_config)
        graph.apply(concept_kb)

    main(args, parser, concept_kb=concept_kb)