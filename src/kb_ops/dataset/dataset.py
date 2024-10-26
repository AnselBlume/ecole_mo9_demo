import os
import pickle
from torch.utils.data import Dataset
from image_processing import LocalizerAndSegmenter
from image_processing.localize_and_segment import LocalizeAndSegmentOutput
from kb_ops.caching import CachedImageFeatures
from kb_ops.train_test_split import split_from_paths
from PIL import Image
from tqdm import tqdm
from model.concept import ConceptKB, ConceptExample
from typing import Optional
import logging
from kb_ops.concurrency import load_pickle, PathToLockMapping
from utils import open_image

logger = logging.getLogger(__file__)

NEGATIVE_LABEL = '[NEGATIVE_LABEL]'

class BaseDataset(Dataset):
    NEGATIVE_LABEL = NEGATIVE_LABEL

    def __init__(
        self,
        data: list,
        labels: list[str],
        concepts_to_train_per_example: list[list[Optional[str]]] = None,
        train_all_concepts_if_unspecified: bool = False,
        path_to_lock: PathToLockMapping = None
    ):
        '''
            concepts_to_train_per_example: List of length n_examples of lists of concept names to train for each example.
                None for an example indicates all concepts should be trained for that example.

                Passing in None as the list of lists (the concepts_to_train_per_example object) will result in all concepts being trained for all examples
                if train_all_concepts_if_unspecified is True.
                Otherwise, if concepts_to_train_per_example is None and train_all_concepts_if_unspecified is False, only the positive concept will be trained for each example.

            train_all_concepts_if_unspecified: If True, all concepts will be trained for all examples if concepts_to_train_per_example is None.
                Otherwise, only the positive concept will be trained for each example if concepts_to_train_per_example is None.

            path_to_lock: Mapping from paths to locks for the data. If provided, the lock will be acquired before loading the data and released after.
        '''
        if not concepts_to_train_per_example:
            logger.debug('concepts_to_train_per_example not provided for dataset; constructing')
            concepts_to_train_per_example = self.get_concepts_to_train_per_example(labels, train_all_concepts_if_unspecified)

        assert len(data) == len(labels) == len(concepts_to_train_per_example)

        # Make a copy of the lists, as these may be modified by extension
        self.data = list(data)
        self.labels = list(labels)
        self.concepts_to_train_per_example = list(concepts_to_train_per_example)
        self.path_to_lock = path_to_lock

    def extend(self, data: list, labels: list[str], concepts_to_train_per_example: list[list[str]] = None, train_all_concepts_if_unspecified: bool = False):
        '''
            Extends the dataset with new data, labels, and concepts_to_train_per_example.
        '''
        if not concepts_to_train_per_example:
            logger.debug('concepts_to_train_per_example not provided for dataset; constructing')
            concepts_to_train_per_example = self.get_concepts_to_train_per_example(labels, train_all_concepts=train_all_concepts_if_unspecified)

        assert len(data) == len(labels) == len(concepts_to_train_per_example)

        self.data.extend(data)
        self.labels.extend(labels)
        self.concepts_to_train_per_example.extend(concepts_to_train_per_example)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_concepts_to_train_per_example(labels: list[str], train_all_concepts: bool):
        '''
            Generates the concepts_to_train_per_example object for a dataset based on the labels and the train_all_concepts flag.
            Specifies for each example which concepts should be trained. None indicates all concepts should be trained for that example.

            If train_all_concepts is True, all concepts will be trained for all examples. Otherwise, only the positive concept will be trained for each example.
        '''
        if train_all_concepts:
            logger.debug('train_all_concepts is True; constructing dataset which trains all concepts for all examples.')
            concepts_to_train_per_example = [None for _ in range(len(labels))]

        else:
            logger.debug('train_all_concepts is False; constructing dataset which trains only positive concept for each example.')
            if any(label == NEGATIVE_LABEL for label in labels):
                raise RuntimeError('Negative labels found in dataset; cannot construct concepts_to_train_per_example automatically.')

            concepts_to_train_per_example = [[label] for label in labels]

        return concepts_to_train_per_example

    def get_metadata(self, index: int):
        return {
            'index': index,
            'label': self.labels[index],
            'concepts_to_train': self.concepts_to_train_per_example[index]
        }

class ImageDataset(BaseDataset):
    def __init__(
        self,
        img_paths: list[str],
        labels: list[str],
        concepts_to_train_per_example: list[list[str]] = None,
        train_all_concepts_if_unspecified: bool = False,
        path_to_lock: PathToLockMapping = None
    ):
        super().__init__(
            data=img_paths,
            labels=labels,
            concepts_to_train_per_example=concepts_to_train_per_example,
            train_all_concepts_if_unspecified=train_all_concepts_if_unspecified,
            path_to_lock=path_to_lock
        )
        self.img_paths = self.data

    def __getitem__(self, idx):
        img = open_image(self.img_paths[idx])
        label = self.labels[idx]
        concepts_to_train = self.concepts_to_train_per_example[idx]

        return {
            'index': idx,
            'image': img,
            'label': label,
            'concepts_to_train': concepts_to_train
        }

'''
    # TODO Make this more space efficient by
    not saving the crops with return_crops, but generating them in the dataset below
'''
class PresegmentedDataset(BaseDataset):
    '''
        Dataset for preprocessed images. segmentation_paths should be the paths to the image segmentations
        corresponding to the labels.
    '''
    def __init__(
        self,
        segmentation_paths: list[str],
        labels: list[str],
        concepts_to_train_per_example: list[list[str]] = None,
        train_all_concepts_if_unspecified: bool = False,
        path_to_lock: PathToLockMapping = None
    ):
        super().__init__(
            data=segmentation_paths,
            labels=labels,
            concepts_to_train_per_example=concepts_to_train_per_example,
            train_all_concepts_if_unspecified=train_all_concepts_if_unspecified,
            path_to_lock=path_to_lock
        )
        self.segmentation_paths = self.data

    def __getitem__(self, idx):
        segmentations: LocalizeAndSegmentOutput = load_pickle(self.segmentation_paths[idx], path_to_lock=self.path_to_lock)
        segmentations.input_image = Image.open(segmentations.input_image_path)

        label = self.labels[idx]
        concepts_to_train = self.concepts_to_train_per_example[idx]

        return {
            'index': idx,
            'segmentations': segmentations,
            'label': label,
            'concepts_to_train': concepts_to_train
        }

class FeatureDataset(BaseDataset):
    def __init__(
        self,
        feature_paths: list[str],
        labels: list[str],
        concepts_to_train_per_example: list[list[str]] = None,
        train_all_concepts_if_unspecified: bool = False,
        path_to_lock: PathToLockMapping = None
    ):
        super().__init__(
            data=feature_paths,
            labels=labels,
            concepts_to_train_per_example=concepts_to_train_per_example,
            train_all_concepts_if_unspecified=train_all_concepts_if_unspecified,
            path_to_lock=path_to_lock
        )
        self.feature_paths = self.data

    def __getitem__(self, idx):
        features: CachedImageFeatures = load_pickle(self.feature_paths[idx], path_to_lock=self.path_to_lock)

        label = self.labels[idx]
        concepts_to_train = self.concepts_to_train_per_example[idx]

        return {
            'index': idx,
            'features': features,
            'label': label,
            'concepts_to_train': concepts_to_train
        }

def split_from_concept_kb(
    concept_kb: ConceptKB,
    split: tuple[float,float,float] = (.6, .2, .2),
    use_concepts_as_negatives: bool = False
):
    '''
        use_concepts_as_negatives: If true, uses each example as a negative for all concepts.
    '''
    # Gather all positive and concept-specific (local) negative feature paths
    pos_feature_paths = []
    neg_feature_paths = [] # Concept-specific
    negs_to_train = []

    for concept in concept_kb:
        for example in concept.examples:
            feature_path = example.image_features_path

            if not feature_path:
                raise RuntimeError(
                    f'Feature path not set for example {example} of concept {concept.name}.'
                    + '\nMake sure to call ConceptKBFeatureCacher.cache_features() before splitting.'
                )

            if example.is_negative:
                neg_feature_paths.append(feature_path)
                negs_to_train.append([concept.name]) # For concept-specific negatives, only train this concept

            else:
                pos_feature_paths.append(feature_path)

    # Split the positive examples
    (trn_ps, trn_ls), (val_ps, val_ls), (tst_ps, tst_ls) = split_from_paths(pos_feature_paths, split=split)

    train_ds = FeatureDataset(trn_ps, trn_ls, train_all_concepts_if_unspecified=use_concepts_as_negatives)
    val_ds = FeatureDataset(val_ps, val_ls)
    test_ds = FeatureDataset(tst_ps, tst_ls)

    # Add concept-specific negatives to train_ds
    neg_labels = [NEGATIVE_LABEL for _ in neg_feature_paths]
    train_ds.extend(neg_feature_paths, neg_labels, negs_to_train)

    # Add global negatives to train_ds
    extend_with_global_negatives(train_ds, concept_kb.global_negatives)

    return train_ds, val_ds, test_ds

def extend_with_global_negatives(ds: FeatureDataset, global_negatives: list[ConceptExample]):
    '''
        Extends a dataset with global negatives.
    '''
    paths = [example.image_features_path for example in global_negatives]
    labels = [NEGATIVE_LABEL for _ in global_negatives]

    ds.extend(paths, labels, train_all_concepts_if_unspecified=True)

def preprocess_segmentations(img_dir: str, out_dir: str, loc_and_seg: LocalizerAndSegmenter):
    '''
        Preprocesses segmentations for a directory of images and saves them to a directory.
        If a segmentation file already exists, it will be skipped.
    '''
    os.makedirs(out_dir, exist_ok=True)

    for img_path in tqdm(os.listdir(img_dir)):
        ext = os.path.splitext(img_path)[1]

        in_path = os.path.join(img_dir, img_path)
        out_path = os.path.join(out_dir, img_path).replace(ext, '.pkl')

        # if os.path.exists(out_path) or ext not in ['.jpg', '.png']:
        #     continue

        img = open_image(in_path)
        segmentations = loc_and_seg.localize_and_segment(img)
        segmentations.input_image_path = in_path

        with open(out_path, 'wb') as f:
            pickle.dump(segmentations, f)

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    from feature_extraction import (
        build_desco,
        build_sam,
    )
    from image_processing import build_localizer_and_segmenter

    in_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/xiaomeng_augmented_data'
    out_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/xiaomeng_augmented_data_segmentations'

    loc_and_seg = build_localizer_and_segmenter(build_sam(), build_desco())

    preprocess_segmentations(in_dir, out_dir, loc_and_seg)