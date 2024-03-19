import argparse
import os
import torch
import h5py
import faiss
import json
import numpy as np
import PIL.Image as Image
import logging

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from feature_extraction.dino_features import DinoFeatureExtractor, get_dino_transform
from utils import open_image, replace_extension

logger = logging.getLogger(__name__)


# TODO: Replace the 'ImageNetPathDataset' from the one in 'kb_ops.dataset.py' - maskrcnn env set up error
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


def reconstruct_img_path(parent_dir, img_path):
    return os.path.join(parent_dir, img_path.split("_")[0], img_path)


def main(args):
    '''
    A utility script to cluster ImageNet images via K-means and pick top-10 images closest to each centroid.
    This script uses DINOv2 as the vision encoder and FAISS API for K-means.
    '''
    if "dino" in args.vis_encoder:
        transform = get_dino_transform()

    # Load the dataloader
    dataset = ImageNetPathDataset(root_dir=f'{args.data_dir}', transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load the vision encoders
    if "dino" in args.vis_encoder:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"device : [ {device} ]")
        dino_model = torch.hub.load('facebookresearch/dinov2', args.vis_encoder)  #.to(device)
        encoder = DinoFeatureExtractor(dino_model)
    else:
        pass # TODO Implement case for additional vision encoders

    img2ids, ids2imgs = {}, {}
    img_embeds, img_paths = [], []
    
    logger.info("[ # of images ] : ", len(dataloader))

    # Encode images
    with torch.no_grad():
        # Construct a FAISS index for the image features
        faiss_index_path = os.path.join(args.out_dir, args.faiss_index_path)
        img_tensor_path = os.path.join(args.out_dir, f"{args.vis_encoder}_img_tensor.pt")
        ids2imgs_path = os.path.join(args.out_dir, f"{args.data_dir.split('/')[-1]}_ids2imgs.json")
        if not os.path.exists(faiss_index_path) or not os.path.exists(img_tensor_path) or not os.path.exists(ids2imgs_path):
            for idx, (img, img_path) in enumerate(tqdm(dataloader, desc=f"Encoding images from {args.data_dir.split('/')[-1]}")):
                img_embed, _ = encoder.forward_from_tensor(img)
                img_embeds.append(img_embed)
                img_paths.extend(list(img_path))

            img_tensor = torch.cat(img_embeds, dim=0)
            embed_dim = img_tensor.shape[1]

            # Save 'img_tensor' for K-means clustering
            torch.save(img_tensor, img_tensor_path)

            # Save 'ids2imgs' for indexing
            img2ids = {replace_extension(img_path): idx for idx, img_path in enumerate(img_paths)}
            ids2imgs = {ids: img_path for img_path, ids in img2ids.items()}  # 'ids2imgs' is the key for image embeddings in 'index_mapped'
            with open(ids2imgs_path, "w") as writer:
                json.dump(ids2imgs, writer)

            img_tensor = img_tensor.detach().cpu().numpy()
            print(f"[ img_tensor (shape) : {img_tensor.shape} ]")

            assert img_tensor.shape[0] == len(ids2imgs.keys())

            index = faiss.IndexFlatL2(embed_dim)
            index_mapped = faiss.IndexIDMap(index)
            faiss.normalize_L2(img_tensor)
            index_mapped.add_with_ids(img_tensor, list(ids2imgs.keys()))
            faiss.write_index(index_mapped, faiss_index_path)  # Save the constructed index using faiss
        else:
            logger.info(f"FAISS Index already exists for {faiss_index_path} from {args.vis_encoder}")
            index_mapped = faiss.read_index(faiss_index_path)
            embed_dim = index_mapped.d
            logger.info(f"Loading 'img_tensor' from {img_tensor_path}")
            img_tensor = torch.load(img_tensor_path)
            logger.info(f"Loading 'ids2imgs' from {ids2imgs_path}")
            with open(ids2imgs_path, "r") as reader:
                ids2imgs = json.load(reader)

        num_cluster = args.num_cluster
        num_iter = args.num_iter
        verbose = True
        
        # Save kmeans centroids for replication
        faiss_ncentroid_path = f"subset-whole_unit-100-faiss-n_{num_cluster}-iter_{num_iter}-kcentroid.npy"
        faiss_ncentroid_path = os.path.join(args.out_dir, faiss_ncentroid_path)
        if not os.path.exists(faiss_ncentroid_path):
            # Perform K-means (k = args.num_cluster) clustering on the image features
            kmeans = faiss.Kmeans(embed_dim, num_cluster, niter=num_iter, verbose=verbose)
            kmeans.train(img_tensor)
            with open(faiss_ncentroid_path, "wb") as writer:
                np.save(writer, kmeans.centroids)
            centroids = kmeans.centroids
        else:
            centroids = np.load(faiss_ncentroid_path)  # use with 'kmeans.train(data, init_centroids=centroids)'
    
        # Search for top-k images per centroid
        top_k = args.top_k
        dist, top_k_indices = index_mapped.search(centroids, top_k)
        logger.info("[ (sample) dist ] :: ", dist[:3])
        logger.info("[ (sample) top_k_indices ] :: ", top_k_indices[:3])

        # Save the image paths to the top-k images per centroid
        nearest_img_path = f"nearest_img_paths_top_k_{top_k}-n_{num_cluster}-iter_{num_iter}.json"
        nearest_img_path = os.path.join(args.out_dir, nearest_img_path)
        if not os.path.exists(nearest_img_path):
            nearest_img_dict = {}
            for centroid_idx, img_indices in enumerate(top_k_indices):
                nearest_img_dict[centroid_idx] = [reconstruct_img_path(args.data_dir, ids2imgs[str(img_idx)]) for img_idx in img_indices]

            with open(nearest_img_path, "w") as fp:
                json.dump(nearest_img_dict, fp)
        else:
            with open(nearest_img_path, "r") as fp:
                nearest_img_dict = json.load(fp)
        
        # Try visualizing each image using 'extract_img_features.ipynb'
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unify data formats for Web navigation datasets")

    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--data_dir", type=str, default="/shared/nas2/blume5/fa23/ecole/data/imagenet/subset-whole_unit-100")
    parser.add_argument("--out_dir", type=str, default="./subset-whole_unit-100-index", help="Output dir for FAISS index of image features")
    parser.add_argument("--faiss_index_path", type=str, default="subset-whole_unit-100-faiss.index")
    parser.add_argument("--faiss_ncentroid_path", type=str, default="subset-whole_unit-100-faiss-kcentroid.npy")
    parser.add_argument("--nearest_img_path", type=str, default="nearest_img_path.pkl", help="Image paths to the 'top_k' nearest features for 'n' centroids")
    parser.add_argument("--vis_encoder", type=str, default="dinov2_vits14")
    
    parser.add_argument("--num_cluster", type=int, default=10, help="The number of clusters for K-means clustering")
    parser.add_argument("--num_iter", type=int, default=200, help="The number of iterations for K-means clustering")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K nearest features to each centroid after K-means clustering")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction")
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    main(args)