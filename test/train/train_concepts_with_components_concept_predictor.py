# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
sys.path.append(os.path.realpath(os.path.join(__file__, '../../../src')))
from model.concept import ConceptKB, ConceptKBConfig
from feature_extraction import build_feature_extractor, build_sam, build_desco, build_clip, build_dino
from image_processing import build_localizer_and_segmenter
from kb_ops import ConceptKBFeaturePipeline, ConceptKBFeatureCacher
from feature_extraction.trained_attrs import N_ATTRS_DINO
from kb_ops.build_kb import label_from_directory, kb_from_img_dir, add_global_negatives
from scripts.utils import set_feature_paths
from controller import Controller
from kb_ops import CLIPConceptRetriever
import shutil
import logging, coloredlogs
logger = logging.getLogger(__file__)

coloredlogs.install(level='DEBUG')

def train_in_tandem():
    from scripts.train.train_and_cls import parse_args, main

    cache_root = 'temp_cache_delete_me'
    args, parser = parse_args(config_str=f'''
        img_dir: /shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/march_demo_test
        extract_label_from: directory

        cache:
            root: {cache_root}

        train:
            n_epochs: 3
            limit_global_negatives: 5
    ''')

    concept_kb = kb_from_img_dir(args.img_dir, label_from_path_fn=label_from_directory)

    # Add component concept relations
    tank_components = {
        'cannon': concept_kb['cannon'],
        'track': concept_kb['track'],
    }

    half_track_components = {
        'wheel': concept_kb['wheel']
    }

    concept_kb['tank'].component_concepts.update(tank_components)
    concept_kb['half_track'].component_concepts.update(half_track_components)

    # Try to load segmentations from cache in case they are already computed
    set_feature_paths(concept_kb, segmentations_dir=f'{cache_root}/segmentations')

    main(args, parser, concept_kb=concept_kb)

def train_sequential():
    img_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/march_demo_test'
    negatives_img_dir = '/shared/nas2/blume5/fa23/ecole/data/imagenet/negatives_rand_1k'

    # Remove possible previously cached features so as not to interfere with test
    if os.path.exists('feature_cache/features'):
        shutil.rmtree('feature_cache/features')

    # Create a concept_kb entirely for the purposes of extracting the examples so we don't have to do it manually
    concept_kb = kb_from_img_dir(img_dir, label_from_path_fn=label_from_directory)
    add_global_negatives(concept_kb, negatives_img_dir, limit=3)
    global_negatives = concept_kb.global_negatives

    concept_examples = {
        concept.name : concept.examples
        for concept in concept_kb
    }

    # Start from empty KB with only global negatives
    concept_kb = ConceptKB(global_negatives=global_negatives)

    # %% Build controller components
    # loc_and_seg = build_localizer_and_segmenter(build_sam(), build_desco())
    loc_and_seg = build_localizer_and_segmenter(build_sam(), None) # Save time by not loading DesCo for this debugging
    clip = build_clip()
    feature_extractor = build_feature_extractor(dino_model=build_dino(), clip_model=clip[0], clip_processor=clip[1])
    feature_pipeline = ConceptKBFeaturePipeline(concept_kb, loc_and_seg, feature_extractor)

    retriever = CLIPConceptRetriever(concept_kb.concepts, *clip)
    cacher = ConceptKBFeatureCacher(concept_kb, feature_pipeline)

    # %% Build controller with an empty ConceptKB
    controller = Controller(loc_and_seg, concept_kb, feature_extractor, retriever=retriever, cacher=cacher)

    concept_kb.initialize(ConceptKBConfig(n_trained_attrs=N_ATTRS_DINO), llm_client=controller.llm_client) # There aren't any concepts yet, but we set its config with initialize anyways

    # %% Add concepts components first, and set examples from our automatically constructed ConceptKB
    track = controller.add_concept('track')
    track.examples = concept_examples['track']

    cannon = controller.add_concept('cannon')
    cannon.examples = concept_examples['cannon']

    tank = controller.add_concept('tank', component_concept_names=['track', 'cannon'])
    tank.examples = concept_examples['tank']

    # %% Train concept in isolation
    for concept_name in ['track', 'cannon', 'tank']:
        logger.info(f'Training {concept_name}')
        controller.train_concept(concept_name)

# %%

# %%
if __name__ == '__main__':
    # train_sequential()
    train_in_tandem()