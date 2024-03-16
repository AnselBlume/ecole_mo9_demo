# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
sys.path.append(os.path.realpath(os.path.join(__file__, '../../src')))
from model.concept import ConceptKB, ConceptExample, ConceptKBConfig
from feature_extraction import build_feature_extractor, build_sam, build_desco, build_clip, build_dino
from image_processing import build_localizer_and_segmenter
from kb_ops import ConceptKBFeaturePipeline, ConceptKBFeatureCacher
from feature_extraction.trained_attrs import N_ATTRS_SUBSET
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
        img_feature_dim=feature_extractor.clip.config.projection_dim,
        n_trained_attrs=N_ATTRS_SUBSET,
    ))

    # %% Build controller
    controller = Controller(loc_and_seg, concept_kb, feature_extractor, retriever=retriever, cacher=cacher)

    # %% Add pizza cutter concept
    img_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/pizza_cutters'
    concept_name = 'pizza cutter'
    cache_dir = '/shared/nas2/blume5/fa23/ecole/cache/pizza_cutters'

    img_paths = list_paths(img_dir, exts=['.jpg', '.png'])

    concept = controller.add_concept(concept_name)
    concept.examples = [ConceptExample(image_path=img_path) for img_path in img_paths]

    set_feature_paths([concept], segmentations_dir=cache_dir + '/segmentations')

    # %% Train pizza cutter
    controller.train_concept(concept_name)

    # %% Add dog concept
    img_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/dogs'
    concept_name = 'dog'
    cache_dir = '/shared/nas2/blume5/fa23/ecole/cache/dogs'

    img_paths = list_paths(img_dir, exts=['.jpg', '.png'])

    concept = controller.add_concept(concept_name)
    concept.examples = [ConceptExample(image_path=img_path) for img_path in img_paths]

    set_feature_paths([concept], segmentations_dir=cache_dir + '/segmentations')

    # %% Train concept in isolation
    controller.train_concept(concept_name)

    # %% Predict examples for verification
    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')
        output = controller.predict_concept(img) # Displays image in Jupyter
        logger.info('Predicted label: ' + output['predicted_label'])

    # %% Test the controller's train function to train everything
    controller.train()
# %%
