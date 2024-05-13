# %%
import os
from typing import Iterable
import numpy as np
from kb_ops.build_kb import label_from_path

import logging
logger = logging.getLogger(__name__)

def split_from_directory(
    img_dir: str,
    exts: list[str] = ['.jpg', '.png'],
    label_from_path_fn=label_from_path,
    **split_kwargs
):
    '''
        Splits images in a directory into train, validation, and test sets.

        Arguments:
            img_dir (str): Directory containing images.
            split (tuple[float,float,float]): Proportions for train, validation, and test sets.
            exts (list[str]): List of file extensions to consider.
            stratified (bool): Whether to stratify the split by label.
            label_from_path_fn (Callable): Function to extract label from path.
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
        label_from_path_fn=label_from_path_fn,
        **split_kwargs
    )

def split_from_paths(
    paths: list[str],
    label_from_path_fn=label_from_path,
    **split_kwargs
) -> tuple[tuple[list,list], tuple[list,list], tuple[list,list]]:
    '''
        Splits images in a directory into train, validation, and test sets.

        Arguments:
            img_dir (str): Directory containing images.
            split (tuple[float,float,float]): Proportions for train, validation, and test sets.
            stratified (bool): Whether to stratify the split by label.
            label_from_path_fn (Callable): Function to extract label from path.
            seed (int): Random seed for reproducibility.

        Returns: 3-Tuple of 2-tuples, each containing a list of paths and a list of labels. 2-tuples are returned
            in the order (train, validation, test).
    '''
    paths = sorted(paths)
    labels = [label_from_path_fn(p) for p in paths]

    return split(
        data=paths,
        labels=labels,
        **split_kwargs
    )

def split(
    data: list,
    labels: list,
    split: tuple[float,float,float] = (0.6,0.2,0.2),
    stratified = True,
    seed: int = 42
):
    assert len(data) == len(labels)

    if stratified:
        tr_ps, v_ps, te_ps = [], [], []
        tr_ls, v_ls, te_ls = [], [], []

        label_to_data = {}
        for datum, label in zip(data, labels):
            label_to_data.setdefault(label, []).append(datum)

        for label, label_data in label_to_data.items():
            tr, v, te = train_val_test_split(label_data, split, seed)
            tr_ps.extend(tr)
            v_ps.extend(v)
            te_ps.extend(te)

            tr_ls.extend([label] * len(tr))
            v_ls.extend([label] * len(v))
            te_ls.extend([label] * len(te))

        # Shuffle each split
        rng = np.random.default_rng(seed)

        tr_perm = rng.permutation(len(tr_ps))
        tr_ps = [tr_ps[i] for i in tr_perm]
        tr_ls = [tr_ls[i] for i in tr_perm]

        v_perm = rng.permutation(len(v_ps))
        v_ps = [v_ps[i] for i in v_perm]
        v_ls = [v_ls[i] for i in v_perm]

        te_perm = rng.permutation(len(te_ps))
        te_ps = [te_ps[i] for i in te_perm]
        te_ls = [te_ls[i] for i in te_perm]

    else:
        tr_ps, v_ps, te_ps = train_val_test_split(data, split, seed)
        tr_ls, v_ls, te_ls = train_val_test_split(labels, split, seed)

    return (tr_ps, tr_ls), (v_ps, v_ls), (te_ps, te_ls)

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