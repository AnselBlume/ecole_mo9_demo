# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Both of these paths are needed here: ../.. for this file to import from src.**, and ../../src for the files in the src
# directory. Technically, we could remove the src. prefix in imports here, but then VSCode analyzer fails
import sys
sys.path.append(os.path.realpath(os.path.join(__file__, '../..'))) # Both of the following are needed here, in o
sys.path.append(os.path.realpath(os.path.join(__file__, '../../src')))
from src.model.concept import ConceptKB, ConceptExample
from src.feature_extraction import build_feature_extractor, build_sam, build_desco, build_clip
from src.image_processing import build_localizer_and_segmenter
from src.kb_ops import ConceptKBFeaturePipeline, ConceptKBFeatureCacher
from src.controller import Controller
from src.kb_ops import CLIPConceptRetriever
from src.scripts.utils import set_feature_paths
import matplotlib.pyplot as plt
from PIL import Image
import logging, coloredlogs
logger = logging.getLogger(__file__)
coloredlogs.install(level='INFO')

# %%
if __name__ == '__main__':
    # Load ConceptKB
    ckpt_path = '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_03_03-02:51:06-n8qy42lu-lr_.01/concept_kb_epoch_15.pt'
    concept_kb = ConceptKB.load(ckpt_path)

    # %% Build controller components
    # loc_and_seg = build_localizer_and_segmenter(build_sam(), build_desco())
    loc_and_seg = build_localizer_and_segmenter(build_sam(), None)
    clip = build_clip()
    feature_extractor = build_feature_extractor(*clip)
    feature_pipeline = ConceptKBFeaturePipeline(concept_kb, loc_and_seg, feature_extractor)

    retriever = CLIPConceptRetriever(concept_kb.concepts, *clip)

    # %% Build controller
    controller = Controller(loc_and_seg, concept_kb, feature_extractor, retriever=retriever)

    # %% Prepare concept for training
    img_paths = [f'/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/dogs/dog_{i}.png' for i in range(1,3)]
    cache_dir = '/shared/nas2/blume5/fa23/ecole/cache/dogs'

    # %%
    controller.add_concept('dog')
    concept = controller.retrieve_concept('dog')
    concept.examples = [ConceptExample(image_path=img_path) for img_path in img_paths]

    # %% Generate segmentations, features for new examples
    cacher = ConceptKBFeatureCacher(concept_kb, feature_pipeline, cache_dir=cache_dir)
    cacher.cache_segmentations()
    cacher.cache_features()

    # %% Train concept in isolation
    controller.train_concept('dog', until_correct_examples=concept.examples)
    logger.info('Finished training new concept')

    # %% Predict examples for verification
    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')
        output = controller.predict_concept(img) # Displays image in Jupyter