from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, img_paths: list[str], labels: list[str]):
        assert len(img_paths) == len(labels)

        self.img_paths = img_paths
        self.labels = labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        label = self.labels[idx]

        return {
            'image': img,
            'label': label
        }