import os
from model.concept import Concept, ConceptKB, ConceptExample
from typing import Callable
import numpy as np
import json
import torch
import pycocotools.mask as mask_utils
import logging

logger = logging.getLogger(__file__)

# Path to data file mapping concepts to required and likely attributes in the same format as returned
# by the LLM in attr_retrieval.retrieve_attributes method
CONCEPT_TO_ATTRS_PATH = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/jx_jsons/union.json'

def label_from_path(path):
    return os.path.basename(path).split('_')[0].lower()

def label_from_directory(path):
    return os.path.basename(os.path.dirname(path)).lower()

def list_paths(
    root_dir: str,
    exts: list[str] = None,
    follow_links: bool = True
):
    '''
        Lists all files in a directory with a given extension.

        Arguments:
            root_dir (str): Directory to search.
            exts (list[str]): List of file extensions to consider.

        Returns: List of paths.
    '''
    exts = set(exts) if exts else None
    paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=follow_links):
        for filename in filenames:
            path = os.path.join(dirpath, filename)

            if exts and os.path.splitext(path)[1].lower() in exts:
                paths.append(path)

    paths = sorted(paths)

    return paths

def kb_from_img_dir(
    img_dir: str,
    label_from_path_fn: Callable[[str],str] = label_from_path,
    exts: list[str] = ['.jpg', '.jpeg', '.webp', '.png'],
    follow_links: bool = True
) -> ConceptKB:
    '''
        Constructs a concept knowledge base from images in a directory.

        Arguments:
            img_dir (str): Directory containing images.

        Returns: ConceptKB
    '''
    kb = ConceptKB()

    for path in list_paths(img_dir, exts=exts, follow_links=follow_links):
        label = label_from_path_fn(path)

        if label not in kb:
            kb.add_concept(Concept(label))

        kb.get_concept(label).examples.append(ConceptExample(concept_name=label, image_path=path))

    return kb

def kb_from_img_and_mask_dirs(
    img_dir: str,
    mask_dir: str,
    label_from_path_fn: Callable[[str],str] = label_from_path,
    exts: list[str] = ['.jpg', '.jpeg', '.webp', '.png'],
    follow_links: bool = True,
    include_images_without_root_concepts: bool = True
) -> ConceptKB:
    '''
        Constructs a concept knowledge base from images in a directory.

        Arguments:
            img_dir (str): Directory containing images.

        Returns: ConceptKB
    '''
    kb = ConceptKB()

    def validate_rle_dict(rle_dict: dict):
        '''
            Validates an RLE dict by checking for the presence of required keys.
        '''
        keys = sorted(list(rle_dict.keys()))

        if set(keys) != {'counts', 'size', 'image_path', 'is_root_concept'}:
            raise ValueError(
                f'Invalid RLE dict. Expected keys: {sorted(["counts", "size", "image_path", "is_root_concept"])}. '
                + f'Got: {keys}'
            )

    def add_example_to_concept(label: str, example: ConceptExample):
        '''
            Adds a ConceptExample to a Concept whose name/label is provided.
        '''

        if label not in kb:
            kb.add_concept(Concept(label))

        kb.get_concept(label).examples.append(example)

    images_with_root_concept_annotations: set[str] = set()

    # Construct ConceptExamples from masks
    for mask_path in list_paths(mask_dir, exts=['.json'], follow_links=follow_links):
        with open(mask_path, 'r') as f:
            rle_dict = json.load(f)

        validate_rle_dict(rle_dict)

        label = label_from_path_fn(mask_path)
        image_path = rle_dict['image_path']

        example = ConceptExample(
            concept_name=label,
            image_path=image_path,
            object_mask_rle_json_path=mask_path
        )

        add_example_to_concept(label, example)

        # Store whether image has a root mask
        if rle_dict['is_root_concept']:
            images_with_root_concept_annotations.add(image_path)

    # Potentially create ConceptExamples for images without root concept masks
    if include_images_without_root_concepts:
        for image_path in list_paths(img_dir, exts=exts, follow_links=follow_links):
            if image_path in images_with_root_concept_annotations:
                continue

            label = label_from_path_fn(image_path)
            example = ConceptExample(concept_name=label, image_path=image_path) # No mask
            add_example_to_concept(label, example)

    return kb

def add_global_negatives(
    concept_kb: ConceptKB,
    img_dir: str,
    exts: list[str] = ['.jpg', '.png'],
    limit: int = None,
    shuffle_paths: bool = True,
    shuffle_seed: int = 42
):
    '''
        Adds global negatives to a concept knowledge base.

        Arguments:
            concept_kb (ConceptKB): Concept knowledge base.
            img_dir (str): Directory containing images.
    '''
    paths = list_paths(img_dir, exts=exts)

    if shuffle_paths:
        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(paths)

    paths = paths[:limit] # [:None] is a nop

    for path in paths:
        concept_kb.global_negatives.append(
            ConceptExample(image_path=path, is_negative=True)
        )

def add_object_masks(
    concept_examples: list[ConceptExample],
    img_dir: str,
    json_rle_mask_dir: str,
    load_full_masks: bool = False
):
    '''
        Adds object masks to concept examples. This assumes there is a one-to-one correspondence
        between images and mask JSON files.

        If there are multiple masks per image, one should use the kb_from_img_and_mask_dirs function.

        Arguments:
            concept_examples (list[ConceptExample]): Concept examples.
            img_dir (str): Directory containing images.
            json_rle_mask_dir (str): Directory containing JSON-encoded object masks.
            load_full_masks (bool): Whether to load full masks along with the mask paths.
    '''
    for example in concept_examples:
        rel_path = os.path.relpath(example.image_path, img_dir)
        mask_rel_path = os.path.splitext(rel_path)[0] + '.json'
        mask_path = os.path.join(json_rle_mask_dir, mask_rel_path)

        if not os.path.exists(mask_path):
            logger.warning(f'Mask file for image {example.image_path} not found at {mask_path}')

        example.object_mask_rle_json_path = mask_path
        if load_full_masks:
            with open(mask_path, 'r') as f:
                rle_dict = json.load(f)
            example.object_mask = torch.from_numpy(mask_utils.decode(rle_dict)).bool()