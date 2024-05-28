# %%
import os
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
sys.path.append(os.path.realpath(os.path.join(__file__, '../../..')))
from model.concept import ConceptKB, ConceptExample, ConceptKBConfig
from feature_extraction import build_feature_extractor, build_sam, build_desco, build_clip, build_dino
from image_processing import build_localizer_and_segmenter
from kb_ops import ConceptKBFeaturePipeline, ConceptKBFeatureCacher, add_global_negatives
from feature_extraction.trained_attrs import N_ATTRS_DINO
from controller import Controller
from kb_ops import CLIPConceptRetriever
from scripts.utils import set_feature_paths
from kb_ops.build_kb import list_paths
from scripts.vis_dino.predictor_heatmap import vis_concept_predictor_heatmap
import logging, coloredlogs
logger = logging.getLogger(__file__)

def prepare_concept(img_dir: str, concept_name: str, cache_dir: str, controller: Controller, set_segmentation_paths: bool = True):
    img_paths = list_paths(img_dir, exts=['.jpg', '.png'])

    concept = controller.add_concept(concept_name)
    concept.examples = [ConceptExample(image_path=img_path) for img_path in img_paths]

    if set_segmentation_paths:
        set_feature_paths([concept], segmentations_dir=cache_dir + '/segmentations')

    return img_paths

if __name__ == '__main__':
    # Setup
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

    add_global_negatives(concept_kb, negatives_img_dir, limit=250)
    set_feature_paths(concept_kb.global_negatives, segmentations_dir=negatives_seg_dir)

    # %% Build controller
    controller = Controller(loc_and_seg, concept_kb, feature_extractor, retriever=retriever, cacher=cacher)

    # %% Set target image for visualization that will not be trained on
    target_img_path = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/march_demo_test/tank/m1_abrams.jpg'

    # %% Prepare tank
    img_dir1 = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/march_demo_test/tank'
    concept1_name = 'tank'
    cache_dir1 = '/shared/nas2/blume5/fa23/ecole/cache/tank'

    prepare_concept(img_dir1, concept1_name, cache_dir1, controller, set_segmentation_paths=False)
    concept1 = controller.retrieve_concept(concept1_name) # Exclude target image
    concept1.examples = [e for e in concept1.examples if e.image_path != target_img_path]

    # %% Train tank
    controller.train_concept(concept1_name, n_epochs=20)

    # %% Prepare track
    img_dir2 = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/march_demo_test/track'
    concept2_name = 'track'
    cache_dir2 = '/shared/nas2/blume5/fa23/ecole/cache/track'

    prepare_concept(img_dir2, concept2_name, cache_dir2, controller, set_segmentation_paths=False)
    concept2 = controller.retrieve_concept(concept2_name) # Exclude target image
    concept2.examples = [e for e in concept2.examples if e.image_path != target_img_path]

    # %% Train track
    controller.train_concept(concept2_name, n_epochs=20)

    # %% Visualize track cls heatmap over tank image
    strategy = 'clamp'

    fig1, _ = vis_concept_predictor_heatmap(
        concept1,
        target_img_path,
        controller.feature_pipeline.feature_extractor.dino_feature_extractor,
        strategy=strategy
    )

    fig2, _ = vis_concept_predictor_heatmap(
        concept2,
        target_img_path,
        controller.feature_pipeline.feature_extractor.dino_feature_extractor,
        strategy=strategy
    )
# %%
