# %%
'''
    Script to visualize the evolution of classifier heatmaps after training a concept one example
    at a time.
'''
import os
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys
sys.path.append(os.path.realpath(os.path.join(__file__, '../../..'))) # src
sys.path.append(os.path.realpath(os.path.join(__file__, '../../../../test'))) # For test.train_from_scratch
from model.concept import ConceptKB, ConceptExample, ConceptKBConfig, Concept
from feature_extraction import build_feature_extractor, build_sam, build_desco, build_clip, build_dino
import torch
from train_from_scratch import prepare_concept # in test module
from image_processing import build_localizer_and_segmenter
from kb_ops import ConceptKBFeaturePipeline, ConceptKBFeatureCacher
from feature_extraction.trained_attrs import N_ATTRS_SUBSET
from controller import Controller
from kb_ops import CLIPConceptRetriever
from scripts.utils import set_feature_paths
from kb_ops.build_kb import list_paths
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from gen_dino_feats import build_dino, DinoFeatureExtractor, get_rescaled_features
from cls_heatmap import normalize
import logging, coloredlogs
from rembg import remove, new_session

logger = logging.getLogger(__file__)

coloredlogs.install(level='DEBUG')

def vis_concept_predictor_heatmap(
    concept: Concept,
    img_path: str,
    fe: DinoFeatureExtractor,
    figsize=(15, 10),
    title=None
):
    # Image and image predictor
    img = Image.open(img_path).convert('RGB')
    img = remove(img, post_process_mask=True, session=new_session('isnet-general-use')).convert('RGB')

    feature_predictor = nn.Sequential(
        concept.predictor.img_features_predictor,
        concept.predictor.img_features_weight
    ).eval().cpu()

    # Patch features
    cls_feats, patch_feats = get_rescaled_features(fe, img, resize_image=False, interpolate_on_cpu=True)
    patch_feats = patch_feats[0] # (h, w, d)

    # Get heatmap
    with torch.no_grad():
        # Need to move to CPU otherwise runs out of GPU mem on big images
        cum_score = feature_predictor(cls_feats.cpu()).item()
        heatmap = feature_predictor(patch_feats.cpu()).squeeze() # (h, w)

    # Move img_features_predictor back to correct device (train is called by train method)
    feature_predictor.cuda()

    heatmap = normalize(heatmap)

    # Visualize heatmap
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(heatmap, cmap='rainbow')
    axs[1].set_title(f'Predictor Heatmap (Score: {cum_score * 100:.2f})')
    axs[1].axis('off')

    if title:
        fig.suptitle(title)

    return fig, axs

# %%
if __name__ == '__main__':
    pass
    # %% Build controller components
    concept_kb = ConceptKB()

    # loc_and_seg = build_localizer_and_segmenter(build_sam(), build_desco())
    loc_and_seg = build_localizer_and_segmenter(build_sam(), None) # Save time by not loading DesCo for this debugging
    clip = build_clip()
    feature_extractor = build_feature_extractor(dino_model=build_dino(), clip_model=clip[0], clip_processor=clip[1])
    dino_fe = feature_extractor.dino_feature_extractor
    feature_pipeline = ConceptKBFeaturePipeline(concept_kb, loc_and_seg, feature_extractor)

    retriever = CLIPConceptRetriever(concept_kb.concepts, *clip)
    cacher = ConceptKBFeatureCacher(concept_kb, feature_pipeline)

    # %% Initialize ConceptKB
    concept_kb.initialize(ConceptKBConfig(
        n_trained_attrs=N_ATTRS_SUBSET,
    ))

    # %% Build controller
    controller = Controller(loc_and_seg, concept_kb, feature_extractor, retriever=retriever, cacher=cacher)

    # %% Add first concept and train
    img_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/single_concepts/hoes'
    concept1_name = 'hoe'
    cache_dir = '/shared/nas2/blume5/fa23/ecole/cache/hoes'

    prepare_concept(img_dir, concept1_name, cache_dir, controller)
    concept1 = controller.retrieve_concept(concept1_name)
    controller.train_concept(concept1_name)

    # %% Add second concept
    img_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/single_concepts/shovels'
    concept2_name = 'shovel'
    cache_dir = '/shared/nas2/blume5/fa23/ecole/cache/shovels'

    img_paths = prepare_concept(img_dir, concept2_name, cache_dir, controller)

    # %% Train second concept, visualizing each
    controller.train_concept(concept2_name)
    img_paths = list_paths(img_dir, exts=['.jpg', '.png'])
    concept2 = controller.add_concept(concept2_name)

    for i, img_path in enumerate(img_paths, start=1):
        new_example = ConceptExample(image_path=img_path)
        concept2.examples.append(new_example)
        set_feature_paths([concept2], segmentations_dir=cache_dir + '/segmentations')

        # %% Visualize, train, and visualize concept 2
        fig, axs = vis_concept_predictor_heatmap(
            concept2,
            img_path,
            dino_fe,
            title=f'{concept2_name.capitalize()} Predictor Heatmap Before Image {i}'
        )
        fig.savefig(f' {concept2_name}_heatmap_before_image_{i}.jpg')

        controller.train_concept(concept2_name, until_correct_examples=[concept2.examples[-1]])

        fig, axs = vis_concept_predictor_heatmap(
            concept2,
            img_path,
            dino_fe,
            title=f'{concept2_name.capitalize()} Predictor Heatmap After Image {i}'
        )
        fig.savefig(f'{concept2_name}_heatmap_after_image_{i}.jpg')

        # %% Visualize, train, and visualize concept 1
        fig, axs = vis_concept_predictor_heatmap(
            concept1,
            img_path,
            dino_fe,
            title=f'{concept1_name.capitalize()} Predictor Heatmap Before Image {i}'
        )
        fig.savefig(f'{concept1_name}_heatmap_before_image_{i}')

        controller.train_concept(concept1_name, stopping_condition='n_epochs', n_epochs=3)

        fig, axs = vis_concept_predictor_heatmap(
            concept1,
            img_path,
            dino_fe,
            title=f'{concept1_name.capitalize()} Predictor Heatmap After Image {i}'
        )
        fig.savefig(f'{concept1_name}_heatmap_after_image_{i}.jpg')

    # %% Predict examples for verification
    # for img_path in img_paths:
    #     img = Image.open(img_path).convert('RGB')
    #     output = controller.predict_concept(img) # Displays image in Jupyter
    #     logger.info('Predicted label: ' + output['predicted_label'])

    # # %% Test the controller's train function to train everything
    # controller.train()
# %%
