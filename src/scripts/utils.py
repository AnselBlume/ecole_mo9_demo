import os
from model.concept import ConceptKB
from datetime import datetime

def get_timestr():
    return datetime.now().strftime('%Y_%m_%d-%H:%M:%S')

def set_feature_paths(concept_kb: ConceptKB, features_dir: str = None, segmentations_dir: str = None):
    '''
        Checks provided segmentation and feature directories for existence. If they exist, sets the
        image_features_path and image_segmentations_path attributes of each example in the concept
        based on the image_path of each example, which is assumed to be set.

        Doing so helps avoid recomputation of features and segmentations in the ConceptKBFeatureCacher,
        which will otherwise recompute these and set the paths in the examples.
    '''
    if features_dir is None and segmentations_dir is None:
        raise ValueError('At least one of features_dir or segmentations_dir must be provided')

    if features_dir and os.path.exists(features_dir):
        # Store pre-computed feature paths in concept examples
        for concept in concept_kb:
            for example in concept.examples:
                basename = os.path.basename(os.path.splitext(example.image_path)[0]) + '.pkl'
                example.image_features_path = os.path.join(features_dir, basename)

    if segmentations_dir and os.path.exists(segmentations_dir):
        # Store presegmented paths in concept examples
        for concept in concept_kb:
            for example in concept.examples:
                basename = os.path.basename(os.path.splitext(example.image_path)[0]) + '.pkl'
                example.image_segmentations_path = os.path.join(segmentations_dir, basename)