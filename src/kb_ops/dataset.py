import os
import pickle
from torch.utils.data import Dataset
from PIL import Image
from controller import Controller
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
        self.data_paths = segmentation_paths
        self.labels = labels

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        with open(self.data_paths[idx], 'rb') as f:
            segmentations = pickle.load(f)

        segmentations['image'] = Image.open(segmentations['image_path'])
        label = self.labels[idx]

        return {
            'segmentations': segmentations,
            'label': label
        }

def preprocess_segmentations(img_dir: str, out_dir: str, controller: Controller):
    '''
        Preprocesses segmentations for a directory of images and saves them to a directory.
        If a segmentation file already exists, it will be skipped.
    '''
    os.makedirs(out_dir, exist_ok=True)

    for img_path in tqdm(os.listdir(img_dir)):
        ext = os.path.splitext(img_path)[1]

        in_path = os.path.join(img_dir, img_path)
        out_path = os.path.join(out_dir, img_path).replace(ext, '.pkl')

        if os.path.exists(out_path) or ext not in ['.jpg', '.png']:
            continue

        img = Image.open(in_path).convert('RGB')
        segmentations = controller.localize_and_segment(img)
        segmentations['image_path'] = in_path

        with open(out_path, 'wb') as f:
            pickle.dump(segmentations, f)

if __name__ == '__main__':
    from feature_extraction import (
        build_desco,
        build_sam,
    )
    from image_processing import build_localizer_and_segmenter

    in_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/xiaomeng_augmented_data'
    out_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/xiaomeng_augmented_data_segmentations'

    controller = Controller(
        build_localizer_and_segmenter(build_sam(), build_desco())
        None
    )

    preprocess_segmentations(in_dir, out_dir, controller)