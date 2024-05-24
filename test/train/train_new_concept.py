# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
sys.path.append(os.path.realpath(os.path.join(__file__, '../../../src')))
from model.concept import ConceptKB, ConceptExample
from feature_extraction import build_feature_extractor, build_sam, build_desco, build_clip, build_dino
from image_processing import build_localizer_and_segmenter
from kb_ops import ConceptKBFeaturePipeline, ConceptKBFeatureCacher
from controller import Controller
from kb_ops import CLIPConceptRetriever
from scripts.utils import set_feature_paths
from kb_ops.build_kb import list_paths
from PIL import Image
import logging, coloredlogs
logger = logging.getLogger(__file__)

coloredlogs.install(level='DEBUG')

# %%
if __name__ == '__main__':
    #  Prepare concept for training
    img_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/single_concepts/pizza_cutters'
    concept_name = 'pizza cutter'
    cache_dir = '/shared/nas2/blume5/fa23/ecole/cache/pizza_cutters'

    # img_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/single_concepts/dogs'
    # concept_name = 'dog'
    # cache_dir = '/shared/nas2/blume5/fa23/ecole/cache/dogs'

    img_paths = list_paths(img_dir, exts=['.jpg', '.png'])

    # %% Load ConceptKB
    ckpt_path = '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_03_23-19:34:45-1qycp7yt-v3-dino_pool_negatives/concept_kb_epoch_20.pt'
    concept_kb = ConceptKB.load(ckpt_path)

    # %% Build controller components
    # loc_and_seg = build_localizer_and_segmenter(build_sam(), build_desco())
    loc_and_seg = build_localizer_and_segmenter(build_sam(), None) # Save time by not loading DesCo for this debugging
    clip = build_clip()
    feature_extractor = build_feature_extractor(dino_model=build_dino(), clip_model=clip[0], clip_processor=clip[1])
    feature_pipeline = ConceptKBFeaturePipeline(concept_kb, loc_and_seg, feature_extractor)

    retriever = CLIPConceptRetriever(concept_kb.concepts, *clip)
    cacher = ConceptKBFeatureCacher(concept_kb, feature_pipeline, cache_dir=cache_dir)

    # %% Build controller
    controller = Controller(loc_and_seg, concept_kb, feature_extractor, retriever=retriever, cacher=cacher)

    # %%
    controller.add_concept(concept_name)
    concept = controller.retrieve_concept(concept_name)
    concept.examples = [ConceptExample(image_path=img_path) for img_path in img_paths]

    # %% Save internal generation time by setting the segmentation paths manually instead of having the controller regenerate them
    # NOTE do not set features here, as they are tied to a particular checkpoint, so changing the checkpoint will change the number of zs features
    set_feature_paths([concept], segmentations_dir=cacher.segmentations_dir)

    # %% Train concept in isolation
    controller.train_concept(concept_name, new_examples=concept.examples)
    logger.info('Finished training new concept')

    # %% Predict examples for verification
    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')
        output = controller.predict_concept(img) # Displays image in Jupyter
# %%
