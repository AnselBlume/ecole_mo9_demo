import os
import pickle
from torch.utils.data import Dataset
from image_processing import LocalizerAndSegmenter
from image_processing.localize_and_segment import LocalizeAndSegmentOutput
from kb_ops.feature_cache import CachedImageFeatures
from PIL import Image
from tqdm import tqdm
from model.concept import ConceptKB, ConceptExample
from typing import Optional
from .train_test_split import split_from_paths

def list_collate(batch):
    keys = batch[0].keys()

    return {k : [d[k] for d in batch] for k in keys}

NEGATIVE_LABEL = '[NEGATIVE_LABEL]'

class BaseDataset(Dataset):
    NEGATIVE_LABEL = NEGATIVE_LABEL

    def __init__(self, data: list, labels: list[str], concepts_to_train: list[list[Optional[str]]] = None):
        '''
            concepts_to_train: List of length n_examples of lists of concept names to train for each example.
                None for an example indicates all concepts should be trained for that example.
                Passing in None as the list of lists will result in all concepts being trained for all examples.
        '''
        if not concepts_to_train:
            concepts_to_train = [None for _ in range(len(data))]

        assert len(data) == len(labels) == len(concepts_to_train)

        self.data = data
        self.labels = labels
        self.concepts_to_train = concepts_to_train

    def extend(self, data: list, labels: list[str], concepts_to_train: list[list[str]] = None):
        if not concepts_to_train:
            concepts_to_train = [None for _ in range(len(data))]

        assert len(data) == len(labels) == len(concepts_to_train)

        self.data.extend(data)
        self.labels.extend(labels)
        self.concepts_to_train.extend(concepts_to_train)

    def __len__(self):
        return len(self.data)

class ImageDataset(BaseDataset):
    def __init__(self, img_paths: list[str], labels: list[str], concepts_to_train: list[list[str]] = None):
        super().__init__(img_paths, labels, concepts_to_train)
        self.img_paths = self.data

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        label = self.labels[idx]
        concepts_to_train = self.concepts_to_train[idx] if self.concepts_to_train else None

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
    def __init__(self, segmentation_paths: list[str], labels: list[str], concepts_to_train: list[list[str]] = None):
        super().__init__(segmentation_paths, labels, concepts_to_train)
        self.segmentation_paths = self.data

    def __getitem__(self, idx):
        with open(self.segmentation_paths[idx], 'rb') as f:
            segmentations: LocalizeAndSegmentOutput = pickle.load(f)

        segmentations.input_image = Image.open(segmentations.input_image_path)
        label = self.labels[idx]
        concepts_to_train = self.concepts_to_train[idx] if self.concepts_to_train else None

        return {
            'index': idx,
            'segmentations': segmentations,
            'label': label,
            'concepts_to_train': concepts_to_train
        }

class FeatureDataset(BaseDataset):
    def __init__(self, feature_paths: list[str], labels: list[str], concepts_to_train: list[list[str]] = None):
        super().__init__(feature_paths, labels, concepts_to_train)
        self.feature_paths = self.data

    def __getitem__(self, idx):
        with open(self.feature_paths[idx], 'rb') as f:
            features: CachedImageFeatures = pickle.load(f)

        label = self.labels[idx]
        concepts_to_train = self.concepts_to_train[idx] if self.concepts_to_train else None

        return {
            'index': idx,
            'features': features,
            'label': label,
            'concepts_to_train': concepts_to_train
        }

def split_from_concept_kb(concept_kb: ConceptKB, split: tuple[float,float,float] = (.6, .2, .2)):
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

    train_ds = FeatureDataset(trn_ps, trn_ls)
    val_ds = FeatureDataset(val_ps, val_ls)
    test_ds = FeatureDataset(tst_ps, tst_ls)

    # Add concept-specific negatives to train_ds
    neg_labels = [NEGATIVE_LABEL for _ in neg_feature_paths]
    train_ds.extend(neg_feature_paths, neg_labels, negs_to_train)

    # Add global engatives to train_ds
    extend_with_global_negatives(train_ds, concept_kb.global_negatives)

    return train_ds, val_ds, test_ds

def extend_with_global_negatives(ds: FeatureDataset, global_negatives: list[ConceptExample]):
    '''
        Extends a dataset with global negatives.
    '''
    paths = [example.image_features_path for example in global_negatives]
    labels = [NEGATIVE_LABEL for _ in global_negatives]

    ds.extend(paths, labels)

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

        img = Image.open(in_path).convert('RGB')
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