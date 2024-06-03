# %%
import os # Change DesCo CUDA device here
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Prepend to path so starts searching at src first
import sys
sys.path = [os.path.join(os.path.dirname(__file__), '../..')] + sys.path

from kb_ops.build_kb import label_from_path, label_from_directory
from kb_ops import kb_from_img_dir
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
        # cache.root: /shared/nas2/blume5/fa23/ecole/cache/airplanes_v1/no_localize

        img_dir: /shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2
        # img_dir: /shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v1_tiny

        train:
            # limit_global_negatives: 5
            split: [.7, .2, 0]
            n_epochs: 50

        extract_label_from: directory

        # XXX This MUST be changed to a directory which we don't care about to not overwrite checkpoints
        cache.root: /shared/nas2/blume5/fa23/ecole/cache/airplanes_v2/rishit_test

        wandb_project: ecole_june_demo_2024

        feature_pipeline_config:
            # Change this to true in order to run component detection with DesCo
            compute_component_concept_scores: false

        loc_and_seg_config:
            do_localize: false

        hierarchy_config:
            concepts:
                - airplane
                - transport plane
                # - cargo jet
                - passenger plane
                # - biplane
                # - fighter jet

                # Component concepts
                - propulsion component
                - wings
                - wing-mounted engine
                # - bulky fuselage
                # - openable nose
                - row of windows
                # - double wings
                # - fixed landing gear
                # - propeller
                # - afterburner

            instance_graph:
                airplane: ['transport plane', 'biplane', 'fighter jet']
                transport plane: ['cargo jet', 'passenger plane']

                propulsion component: ['wing-mounted engine', 'afterburner', 'propeller']
                wings: ['double wings']

            component_graph:
                airplane: ['wings', 'propulsion component']
                transport plane: ['wing-mounted engine']
                cargo jet: ['bulky fuselage', 'openable nose']
                passenger plane: ['row of windows']
                biplane: ['double wings', 'fixed landing gear', 'propeller']
                fighter jet: ['afterburner']
        ''')

    if args.ckpt_path:
        concept_kb = None # Will be loaded in main

    else:
        # Construct KB from images
        label_extractor = label_from_path if args.extract_label_from == 'path' else label_from_directory
        concept_kb = kb_from_img_dir(args.img_dir, label_from_path_fn=label_extractor)

        # Apply graph connectivity and create any concepts without images
        graph = ConceptGraphParser().parse_graph(path=args.hierarchy_config_path, config=args.hierarchy_config)
        graph.apply(concept_kb)

    main(args, parser, concept_kb=concept_kb)