import os
from model.concept import Concept, ConceptKB, ConceptExample
from typing import Callable
import numpy as np

def label_from_path(path):
    return os.path.basename(path).split('_')[0].lower()

def label_from_directory(path):
    return os.path.basename(os.path.dirname(path)).lower()

def list_paths(
    root_dir: str,
    exts: list[str] = None,
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
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            path = os.path.join(dirpath, filename)

            if exts and os.path.splitext(path)[1] in exts:
                paths.append(path)

    paths = sorted(paths)

    return paths

def kb_from_img_dir(
    img_dir: str,
    label_from_path_fn: Callable[[str],str] = label_from_path,
    exts: list[str] = ['.jpg', '.png']
) -> ConceptKB:
    '''
        Constructs a concept knowledge base from images in a directory.

        Arguments:
            img_dir (str): Directory containing images.

        Returns: ConceptKB
    '''
    kb = ConceptKB()

    for path in list_paths(img_dir, exts=exts):
        label = label_from_path_fn(path)

        if label not in kb:
            kb.add_concept(Concept(label))

        kb.get_concept(label).examples.append(ConceptExample(image_path=path))

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