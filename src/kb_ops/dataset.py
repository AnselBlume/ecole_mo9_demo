import os
import pickle
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from image_processing import LocalizerAndSegmenter
from image_processing.localize_and_segment import LocalizeAndSegmentOutput
from kb_ops.feature_cache import CachedImageFeatures
from PIL import Image
from tqdm import tqdm


def list_collate(batch):
    keys = batch[0].keys()

    return {k : [d[k] for d in batch] for k in keys}

class ImageDataset(Dataset):
    def __init__(self, img_paths: list[str], labels: list[str]):
        assert len(img_paths) == len(labels)

        self.img_paths = img_paths
        self.labels = labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        label = self.labels[idx]

        return {
            'index': idx,
            'image': img,
            'label': label
        }

'''
    # TODO Make this more space efficient by
    not saving the crops with return_crops, but generating them in the dataset below
'''
class PresegmentedDataset(Dataset):
    '''
        Dataset for preprocessed images. segmentation_paths should be the paths to the image segmentations
        corresponding to the labels.
    '''
    def __init__(self, segmentation_paths: list[str], labels: list[str]):
        assert len(segmentation_paths) == len(labels)
        self.segmentation_paths = segmentation_paths
        self.labels = labels

    def __len__(self):
        return len(self.segmentation_paths)

    def __getitem__(self, idx):
        with open(self.segmentation_paths[idx], 'rb') as f:
            segmentations: LocalizeAndSegmentOutput = pickle.load(f)

        segmentations.input_image = Image.open(segmentations.input_image_path)
        label = self.labels[idx]

        return {
            'index': idx,
            'segmentations': segmentations,
            'label': label
        }

class FeatureDataset(Dataset):
    def __init__(self, feature_paths: list[str], labels: list[str]):
        assert len(feature_paths) == len(labels)

        self.feature_paths = feature_paths
        self.labels = labels

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        with open(self.feature_paths[idx], 'rb') as f:
            features: CachedImageFeatures = pickle.load(f)

        return {
            'index': idx,
            'features': features,
            'label': self.labels[idx]
        }

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


class ImageNetPathDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Assume each subdirectory in the root directory is a class label
        for label_dir in tqdm(os.listdir(root_dir), desc=f'Initializing image paths for ImageNetPathDataset'):
            label_dir_path = os.path.join(root_dir, label_dir)
            if os.path.isdir(label_dir_path):
                for image_name in os.listdir(label_dir_path):
                    if image_name.endswith('.JPEG'):
                        self.image_paths.append(os.path.join(label_dir_path, image_name))
                        self.labels.append(label_dir)  # Using directory name as label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Convert image to RGB
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, image_path
    

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