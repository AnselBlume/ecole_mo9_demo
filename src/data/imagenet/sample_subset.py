'''
    Sample a subset of ImageNet with a given number of images per class and create
    hard links to the sampled images in a new directory with the same directory structure.
'''
# %%
import os
import pandas as pd
from imagenet_hierarchy import ImageNetConcept, get_hierarchy, dfs
import numpy as np
import coloredlogs, logging
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

def count_train_images(root_node: ImageNetConcept):
    count = 0
    for child in root_node.child_concepts:
        count += count_train_images(child)

    count += root_node.num_train_images

    return count

if __name__ == '__main__':
    # Get the root node: whole, unit
    root_node, concepts = get_hierarchy(return_concepts=True)
    logger.info(f'Number of concepts under root {root_node.words[0]}: {len(concepts)}')
    logger.info(f'Number of training images under root {root_node.words[0]}: {count_train_images(root_node)}')

    # %% Get subtree of new root node, setting new root node and concepts
    target_root_words = 'whole, unit'
    root_node = [c for c in concepts if target_root_words in c.words[0]][0]

    is_reachable = np.zeros(len(concepts), dtype=bool)
    dfs(root_node, concepts, is_reachable)
    concepts = [c for c, reachable in zip(concepts, is_reachable) if reachable]

    logger.info(f'Number of concepts under root {root_node.words[0]}: {len(concepts)}')
    logger.info(f'Number of training images under root {root_node.words[0]}: {count_train_images(root_node)}')

    # %% Sample from imagenet dir
    n_to_sample_per_class = 100

    imagenet_train_dir = '/shared/nas2/blume5/fa23/ecole/data/imagenet/ILSVRC2012_img_train' # Expected to have nested directories of synset ids
    out_dir = '/shared/nas2/blume5/fa23/ecole/data/imagenet/subset-whole_unit-100'
    random_seed = 42

    logger.info(f'Sampling ImageNet with {n_to_sample_per_class} images per class and random seed {random_seed} to {out_dir}')

    os.makedirs(out_dir, exist_ok=True)
    wn_id_to_concept = {c.wordnet_id: c for c in concepts}

    rng = np.random.default_rng(random_seed)
    for wn_id in tqdm(wn_id_to_concept):
        synset_dir = os.path.join(imagenet_train_dir, wn_id)

        if not os.path.exists(synset_dir):
            synset_name = wn_id_to_concept[wn_id].words[0]
            logger.warning(f'{synset_dir} corresponding to {synset_name} does not exist')
            continue

        synset_images = os.listdir(synset_dir)
        synset_images = rng.choice(synset_images, size=n_to_sample_per_class, replace=False)

        # Make output dir with hard links
        os.makedirs(os.path.join(out_dir, wn_id), exist_ok=True)

        for image in synset_images:
            out_path = os.path.join(out_dir, wn_id, image)

            if not os.path.exists(out_path):
                os.symlink(os.path.join(synset_dir, image), out_path)
# %%
