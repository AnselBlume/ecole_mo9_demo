import os
from model.concept import ConceptKB, Concept, ConceptExample
from datetime import datetime
from typing import Union
from itertools import chain

def get_timestr():
    return datetime.now().strftime('%Y_%m_%d-%H:%M:%S')

def set_feature_paths(
    concepts_or_examples: Union[list[Concept], ConceptKB, list[ConceptExample]],
    *,
    segmentations_dir: str = None,
    features_dir: str = None
):
    '''
        Checks provided segmentation and feature directories for existence. If they exist, sets the
        image_features_path and image_segmentations_path attributes of each example in the Concepts
        (or ConceptKB) based on the image_path of each example, which is assumed to be set.

        Doing so helps avoid recomputation of features and segmentations in the ConceptKBFeatureCacher,
        which will otherwise recompute these and set the paths in the examples.
    '''
    if segmentations_dir is None and features_dir is None:
        raise ValueError('At least one of features_dir or segmentations_dir must be provided')

    if isinstance(concepts_or_examples, ConceptKB) or isinstance(concepts_or_examples[0], Concept):
        examples = chain.from_iterable([
            concept.examples for concept in concepts_or_examples
        ])

    else: # list[ConceptExample]
        examples = concepts_or_examples

    def set_paths_if_exists(examples: list[ConceptExample], attr_name: str, base_dir):
        for example in examples:
            base_path = os.path.splitext(os.path.basename(example.image_path))[0] + '.pkl'
            target_path = os.path.join(base_dir, base_path)
            if os.path.exists(target_path):
                setattr(example, attr_name, target_path)

    if segmentations_dir and os.path.exists(segmentations_dir):
        # Store presegmented paths in concept examples
        set_paths_if_exists(examples, 'image_segmentations_path', segmentations_dir)

    if features_dir and os.path.exists(features_dir):
        # Store pre-computed feature paths in concept examples
        set_paths_if_exists(examples, 'image_features_path', features_dir)
