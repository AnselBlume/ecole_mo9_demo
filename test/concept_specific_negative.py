'''
    Script to test concept-specific negatives only training their respective concepts.
'''
# %%
import os
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
sys.path.append(os.path.realpath(os.path.join(__file__, '../../src')))
from model.concept import ConceptKB, ConceptExample, ConceptKBConfig
from feature_extraction import build_feature_extractor, build_sam, build_desco, build_clip, build_dino
from image_processing import build_localizer_and_segmenter
from kb_ops import ConceptKBFeaturePipeline, ConceptKBFeatureCacher, add_global_negatives
from feature_extraction.trained_attrs import N_ATTRS_DINO
from controller import Controller
from kb_ops import CLIPConceptRetriever
from scripts.utils import set_feature_paths
from kb_ops.build_kb import list_paths
from PIL import Image
import logging, coloredlogs
logger = logging.getLogger(__file__)

coloredlogs.install(level='DEBUG')

def prepare_concept(img_dir: str, concept_name: str, cache_dir: str, controller: Controller, set_segmentation_paths: bool = True):
    img_paths = list_paths(img_dir, exts=['.jpg', '.png'])

    concept = controller.add_concept(concept_name)
    concept.examples = [ConceptExample(image_path=img_path) for img_path in img_paths]

    if set_segmentation_paths:
        set_feature_paths([concept], segmentations_dir=cache_dir + '/segmentations')

    return img_paths

# %%
if __name__ == '__main__':
    pass
    # %% Build controller components
    concept_kb = ConceptKB()

    # loc_and_seg = build_localizer_and_segmenter(build_sam(), build_desco())
    loc_and_seg = build_localizer_and_segmenter(build_sam(), None) # Save time by not loading DesCo for this debugging
    clip = build_clip()
    feature_extractor = build_feature_extractor(dino_model=build_dino(), clip_model=clip[0], clip_processor=clip[1])
    feature_pipeline = ConceptKBFeaturePipeline(concept_kb, loc_and_seg, feature_extractor)

    retriever = CLIPConceptRetriever(concept_kb.concepts, *clip)
    cacher = ConceptKBFeatureCacher(concept_kb, feature_pipeline)

    # %% Initialize ConceptKB
    concept_kb.initialize(ConceptKBConfig(
        n_trained_attrs=N_ATTRS_DINO,
    ))

    # Add global negatives
    negatives_img_dir = '/shared/nas2/blume5/fa23/ecole/data/imagenet/negatives_rand_1k'
    negatives_seg_dir = '/shared/nas2/blume5/fa23/ecole/cache/imagenet_rand_1k/segmentations'

    add_global_negatives(concept_kb, negatives_img_dir, limit=10)
    set_feature_paths(concept_kb.global_negatives, segmentations_dir=negatives_seg_dir)

    # %% Build controller
    controller = Controller(loc_and_seg, concept_kb, feature_extractor, retriever=retriever, cacher=cacher)

    # %% Add first concept
    # img_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/single_concepts/pizza_cutters'
    # concept_name = 'pizza cutter'
    # cache_dir = '/shared/nas2/blume5/fa23/ecole/cache/pizza_cutters'
    img_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/single_concepts/hoes'
    concept_name = 'hoe'
    cache_dir = '/shared/nas2/blume5/fa23/ecole/cache/hoes'

    prepare_concept(img_dir, concept_name, cache_dir, controller, set_segmentation_paths=False)

    # %% Add second concept
    # img_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/single_concepts/dogs'
    # concept_name = 'dog'
    # cache_dir = '/shared/nas2/blume5/fa23/ecole/cache/dogs'
    img_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/single_concepts/shovels'
    concept_name = 'shovel'
    cache_dir = '/shared/nas2/blume5/fa23/ecole/cache/shovels'

    img_paths = prepare_concept(img_dir, concept_name, cache_dir, controller, set_segmentation_paths=False)

    # Add a concept-specific negative
    concept_specific_negative_path = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/adversarial_spoon.jpg'
    concept_kb[concept_name].examples.append(ConceptExample(image_path=concept_specific_negative_path, is_negative=True))

    # %% Train second concept
    controller.train_concept(concept_name)

    # %% Test the controller's train function to train everything
    controller.train()
# %%
