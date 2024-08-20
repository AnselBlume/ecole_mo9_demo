# %%
import logging

from .heatmap import ControllerHeatmapMixin
from .interpretation import ControllerInterpretationMixin
from .predict import ControllerPredictionMixin
from .train import ControllerTrainMixin

logger = logging.getLogger(__name__)

class Controller(
    ControllerTrainMixin,
    ControllerPredictionMixin,
    ControllerHeatmapMixin,
    ControllerInterpretationMixin
):
    def set_zs_attributes(self, concept_name: str, zs_attrs: list[str]):
        concept = self.retrieve_concept(concept_name)
        concept.zs_attributes = zs_attrs

        self.cacher.recache_zs_attr_features(concept) # Recompute zero-shot attribute scores
        self.train_concept(concept.name, new_examples=concept.examples)

# %%
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    import coloredlogs
    import PIL
    import PIL.Image
    from feature_extraction import (build_desco, build_feature_extractor,
                                    build_sam)
    from image_processing import build_localizer_and_segmenter
    from kb_ops.feature_pipeline import ConceptKBFeaturePipeline
    from model.concept import ConceptExample, ConceptKB

    coloredlogs.install(level=logging.INFO)

    # %%
    ###############################
    #  June 2024 Demo Checkpoint #
    ###############################
    ckpt_path = '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_06_06-23:31:12-8ckp59v8-all_planes_and_guns/concept_kb_epoch_20.pt'
    kb = ConceptKB.load(ckpt_path)
    loc_and_seg = build_localizer_and_segmenter(build_sam(), None)
    fe = build_feature_extractor()
    feature_pipeline = ConceptKBFeaturePipeline(loc_and_seg, fe)

    controller = Controller(kb, feature_pipeline)

    # %% Add a new concept and train it
    new_concept_name = 'bomber'
    controller.add_concept(new_concept_name, parent_concept_names=['airplane'])

    image_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/bomber'

    examples = [
        ConceptExample(concept_name=new_concept_name, image_path=os.path.join(image_dir, basename))
        for basename in os.listdir(image_dir)
    ]

    controller.add_examples(examples, new_concept_name)
    controller.train_concept(new_concept_name)

    # Add another new concept
    new_concept_name = 'bomber wing'
    # controller.add_concept(new_concept_name, parent_concept_names=['bomber'])
    controller.add_concept(new_concept_name, containing_concept_names=['bomber'])

    image_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/bomber_wings'

    examples = [
        ConceptExample(concept_name=new_concept_name, image_path=os.path.join(image_dir, basename))
        for basename in os.listdir(image_dir)
    ]

    controller.add_examples(examples, new_concept_name)
    controller.train_concepts(['bomber']) # XXX This has a caching bug; must be fixed!

    # %% Run the first prediction
    img_path = '/shared/nas2/blume5/fa23/ecole/Screenshot 2024-06-07 at 6.23.02 AM.png'
    img = PIL.Image.open(img_path).convert('RGB')
    result = controller.is_concept_in_image(img, 'wing-mounted engine', unk_threshold=.001)

    # %%
    result = controller.predict_concept(img, unk_threshold=.1)
    logger.info(f'Predicted label: {result["predicted_label"]}')

    # %% Hierarchical prediction
    result = controller.predict_hierarchical(img, unk_threshold=.1)
    logger.info(f'Concept path: {result["concept_path"]}')

    # %% New concept training
    new_concept_name = 'biplane'
    parent_concept_name = 'airplane'

    new_concept_image_paths = [
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/biplane/000001.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/biplane/000002.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/biplane/000003.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/biplane/000004.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/biplane/000005.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/biplane/000006.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/biplane/000008.jpg'
    ]

    new_concept_examples = [ConceptExample(new_concept_name, image_path=image_path) for image_path in new_concept_image_paths]

    controller.add_concept(new_concept_name, parent_concept_names=[parent_concept_name])

    controller.train_concept(
        new_concept_name,
        new_examples=new_concept_examples
    )

    # %% Predict with new concept
    test_image_paths = [
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/biplane/000010.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/biplane/000011.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/passenger plane/000021.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/fighter jet/Screenshot 2024-06-02 at 9.18.58 PM.png',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/afterburner/sr71.png',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/row of windows/000021.jpg'
    ]
    test_image_labels = [
        'biplane',
        'biplane',
        'passenger plane',
        'fighter jet',
        'afterburner',
        'row of windows'
    ]
    include_component_parts = [
        False,
        False,
        False,
        False,
        True,
        True
    ]

    for img_path, label, include_components in zip(test_image_paths, test_image_labels, include_component_parts):
        img = PIL.Image.open(img_path).convert('RGB')
        result = controller.predict_concept(img, unk_threshold=.1, include_component_concepts=include_components)
        logger.info(f'Predicted label: {result["predicted_label"]}')

        result = controller.predict_hierarchical(img, unk_threshold=.1, include_component_concepts=include_components)
        logger.info(f'Hierarchical Predicted label: {result["predicted_label"]}')

        logger.info(f'Expected label: {label}')

    # %%
    ###############################
    #  March 2024 Demo Checkpoint #
    ###############################
    ckpt_path = '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_03_22-15:06:03-xob6535d-v3-dino_pool/concept_kb_epoch_50.pt'

    kb = ConceptKB.load(ckpt_path)
    loc_and_seg = build_localizer_and_segmenter(build_sam(), build_desco())
    fe = build_feature_extractor()
    feature_pipeline = ConceptKBFeaturePipeline(loc_and_seg, fe)

    controller = Controller(kb, feature_pipeline)

    # %% Run the first prediction
    img_path = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/adversarial_spoon.jpg'
    img = PIL.Image.open(img_path).convert('RGB')
    result = controller.predict_concept(img, unk_threshold=.1)

    logger.info(f'Predicted label: {result["predicted_label"]}')

    # %% Run the second prediction
    img_path2 = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/fork_2_9.jpg'
    img2 = PIL.Image.open(img_path2).convert('RGB')
    controller.predict_concept(img2)

    # %% Explain difference between images
    logger.info('Explaining difference between predictions...')
    controller.compare_predictions(indices=(-2,-1), weight_by_predictors=True)

    # %% Explain difference between image regions
    controller.compare_predictions(indices=(-2,-1), weight_by_predictors=True, image1_regions=[0])
    controller.compare_predictions(indices=(-2,-1), weight_by_predictors=True, image2_regions=[0])
    controller.compare_predictions(indices=(-2,-1), weight_by_predictors=True, image1_regions=[0], image2_regions=[0])

    # %% Explain difference between concepts
    controller.compare_concepts('spoon', 'fork')

    # %% Visualize difference between zero-shot attributes
    controller.compare_zs_attributes(('spoon', 'fork'), img)

    # %%