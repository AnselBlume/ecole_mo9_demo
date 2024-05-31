'''
    Script to train a ConceptKB with a couple of concepts from scratch, one concept after the other.
'''
# %%
import os
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
sys.path.append(os.path.realpath(os.path.join(__file__, '../../../src')))
import pickle
from model.concept import ConceptKB, ConceptExample, ConceptKBConfig
from feature_extraction import build_feature_extractor, build_sam, build_desco, build_clip, build_dino
from image_processing import build_localizer_and_segmenter
from kb_ops import ConceptKBFeaturePipeline, ConceptKBFeatureCacher, add_global_negatives
from feature_extraction.trained_attrs import N_ATTRS_DINO
from kb_ops.caching import CachedImageFeatures
from controller import Controller
from model.attribute import Attribute
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

    add_global_negatives(concept_kb, negatives_img_dir, limit=5)
    set_feature_paths(concept_kb.global_negatives, segmentations_dir=negatives_seg_dir)

    # %% Build controller
    controller = Controller(loc_and_seg, concept_kb, feature_extractor, retriever=retriever, cacher=cacher)

    # %% Add first concept
    img_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/single_concepts/hoes'
    concept_name = 'hoe'
    cache_dir = '/shared/nas2/blume5/fa23/ecole/cache/hoes'

    prepare_concept(img_dir, concept_name, cache_dir, controller, set_segmentation_paths=False)

    # %% Train first concept
    controller.train_concept(concept_name)

    # Count the number of cached zero-shot attributes for an example image
    concept = controller.retrieve_concept(concept_name)

    first_example = concept.examples[0]
    with open(first_example.image_features_path, 'rb') as f:
        image_features: CachedImageFeatures = pickle.load(f)

    orig_n_zs_attrs = image_features.concept_to_zs_attr_img_scores[concept_name].shape[1] # (1, n_zs_attrs)
    logger.info(f'Original number of zs attributes: {orig_n_zs_attrs}')

    # Modify zero-shot attributes
    # NOTE this does NOT test modifying the weights of the ConceptPredictor
    new_attr = Attribute(name='another zero-shot attribute')
    controller.concept_kb[concept_name].zs_attributes.append(new_attr)
    controller.cacher.recache_zs_attr_features(concept, examples=[first_example])

    # Compute the number of zs attributes in the file
    with open(first_example.image_features_path, 'rb') as f:
        image_features: CachedImageFeatures = pickle.load(f)

    new_n_zs_attrs = image_features.concept_to_zs_attr_img_scores[concept_name].shape[1] # (1, n_zs_attrs)
    logger.info(f'New number of zs attributes: {new_n_zs_attrs}')

    assert new_n_zs_attrs == orig_n_zs_attrs + 1, f'Expected {orig_n_zs_attrs + 1} zs attributes, got {new_n_zs_attrs}'
    print('Success!')
