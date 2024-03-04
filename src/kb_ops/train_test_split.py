# %%
import os
from typing import Iterable
import numpy as np
from kb_ops.build_kb import label_from_path

import logging
logger = logging.getLogger(__name__)

def train_val_test_split(
    iterable: Iterable,
    split: tuple[float,float,float] = (0.6,0.2,0.2),
    seed: int = 42
) -> tuple[list,list,list]:
    '''
        Splits an iterable into train, validation, and test sets.

        Arguments:
            iterable (Iterable): Iterable to split.
            split (tuple[float,float,float]): Proportions for train, validation, and test sets.
            seed (int): Random seed for reproducibility.

        Returns: 3-tuple of lists, each containing a split of the iterable. Lists are returned in the order
            (train, validation, test).
    '''
    rng = np.random.default_rng(seed)
    assert np.allclose(sum(split), 1), 'Split values must sum to 1'

    iterable = list(iterable) # Copy
    rng.shuffle(iterable)

    n = len(iterable)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    n_test = n - n_train - n_val

    if n_train == 0 or n_val == 0 or n_test == 0:
        logger.warning(f'A split has zero elements: (train, val, test) = ({n_train}, {n_val}, {n_test})')

    train = iterable[:n_train]
    val = iterable[n_train:n_train+n_val]
    test = iterable[n_train+n_val:]

    return train, val, test

def split_from_directory(
    img_dir: str,
    split: tuple[float,float,float] = (0.6,0.2,0.2),
    exts: list[str] = ['.jpg', '.png'],
    stratified = True,
    label_from_path=label_from_path,
    seed: int = 42
):
    '''
        Splits images in a directory into train, validation, and test sets.

        Arguments:
            img_dir (str): Directory containing images.
            split (tuple[float,float,float]): Proportions for train, validation, and test sets.
            exts (list[str]): List of file extensions to consider.
            stratified (bool): Whether to stratify the split by label.
            label_from_path (Callable): Function to extract label from path.
            seed (int): Random seed for reproducibility.

        Returns: Tuple of 2-tuples, each containing a list of paths and a list of labels. 2-tuples are returned
            in the order (train, validation, test).
    '''
    paths = [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if os.path.splitext(f)[1] in exts
    ]

    return split_from_paths(
        paths=paths,
        split=split,
        stratified=stratified,
        label_from_path=label_from_path,
        seed=seed
    )

def split_from_paths(
    paths: list[str],
    split: tuple[float,float,float] = (0.6,0.2,0.2),
    stratified = True,
    label_from_path=label_from_path,
    seed: int = 42
) -> tuple[tuple[list,list], tuple[list,list], tuple[list,list]]:
    '''
        Splits images in a directory into train, validation, and test sets.

        Arguments:
            img_dir (str): Directory containing images.
            split (tuple[float,float,float]): Proportions for train, validation, and test sets.
            stratified (bool): Whether to stratify the split by label.
            label_from_path (Callable): Function to extract label from path.
            seed (int): Random seed for reproducibility.

        Returns: 3-Tuple of 2-tuples, each containing a list of paths and a list of labels. 2-tuples are returned
            in the order (train, validation, test).
    '''
    paths = sorted(paths)
    labels = [label_from_path(p) for p in paths]

    if stratified:
        tr_p, v_p, te_p = [], [], []
        tr_l, v_l, te_l = [], [], []

        label_to_paths = {}
        for path in paths:
            label = label_from_path(path)
            label_to_paths.setdefault(label, []).append(path)

        for label, label_paths in label_to_paths.items():
            tr, v, te = train_val_test_split(label_paths, split, seed)
            tr_p.extend(tr)
            v_p.extend(v)
            te_p.extend(te)

            tr_l.extend([label] * len(tr))
            v_l.extend([label] * len(v))
            te_l.extend([label] * len(te))

        # Shuffle each split
        rng = np.random.default_rng(seed)

        tr_perm = rng.permutation(len(tr_p))
        tr_p = [tr_p[i] for i in tr_perm]
        tr_l = [tr_l[i] for i in tr_perm]

        v_perm = rng.permutation(len(v_p))
        v_p = [v_p[i] for i in v_perm]
        v_l = [v_l[i] for i in v_perm]

        te_perm = rng.permutation(len(te_p))
        te_p = [te_p[i] for i in te_perm]
        te_l = [te_l[i] for i in te_perm]

    else:
        tr_p, v_p, te_p = train_val_test_split(paths, split, seed)
        tr_l, v_l, te_l = train_val_test_split(labels, split, seed)

    return (tr_p, tr_l), (v_p, v_l), (te_p, te_l)
# %%
