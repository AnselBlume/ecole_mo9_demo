import os
import sys
sys.path = [os.path.realpath(os.path.join(__file__, '../../../../src'))] + sys.path
import PIL
from controller import Controller
from model.concept import ConceptKB
from feature_extraction import build_feature_extractor, build_sam
from image_processing import build_localizer_and_segmenter
from model.concept import ConceptKB
from tqdm import tqdm
import PIL.Image
import coloredlogs
import logging
from kb_ops.feature_pipeline import ConceptKBFeaturePipeline

logger = logging.getLogger(__file__)
coloredlogs.install(level=logging.INFO, logger=logger)

class HeatmapWriter:
    def __init__(self, concept_kb: ConceptKB = None, clamp_center: int = 5):
        self.loc_and_seg = build_localizer_and_segmenter(build_sam(), None)
        self.feature_extractor = build_feature_extractor()
        self.feature_pipeline = ConceptKBFeaturePipeline(self.loc_and_seg, self.feature_extractor)
        self.clamp_center = clamp_center

        if concept_kb:
            self.set_concept_kb(concept_kb)
        else:
            self.controller = None

    def set_concept_kb(self, concept_kb: ConceptKB):
        self.controller = Controller(concept_kb, self.feature_pipeline)
        self.set_clamp_center(self.clamp_center)

    def set_clamp_center(self, clamp_center: int):
        self.clamp_center = clamp_center

        if self.controller:
            self.controller.heatmap_visualizer.config.clamp_center = clamp_center

    def visualize_heatmaps(self, img_path: str, concept_name: str, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)

        img = PIL.Image.open(img_path).convert('RGB')
        concept_names = [concept_name] + list(self.controller.retrieve_concept(concept_name).component_concepts)

        for concept_name in concept_names:
            heatmap = self.controller.heatmap(img, concept_name)
            heatmap.save(os.path.join(out_dir, f'{concept_name}.jpg'))

if __name__ == '__main__':
    ckpt_paths = [
        # No component background removal
        # '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_07_23-00:16:09/concept_kb_epoch_100.pt',
        # '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_07_23-00:16:09/concept_kb_epoch_200.pt',
        # '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_07_23-00:16:09/concept_kb_epoch_300.pt',
        # '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_07_23-00:16:09/concept_kb_epoch_400.pt',

        # With component background removal
        '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_07_23-02:04:37/concept_kb_epoch_100.pt',
        '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_07_23-02:04:37/concept_kb_epoch_200.pt',
        '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_07_23-02:04:37/concept_kb_epoch_300.pt',
        '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_07_23-02:04:37/concept_kb_epoch_400.pt',

        # With component background removal and containing concept training
        '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_07_23-02:35:55/concept_kb_epoch_100.pt',
        '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_07_23-02:35:55/concept_kb_epoch_200.pt',
        '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_07_23-02:35:55/concept_kb_epoch_300.pt',
        '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_07_23-02:35:55/concept_kb_epoch_400.pt',
    ]

    test_img_paths = [
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/2024_july_demo/inference_examples/1-biplane.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/2024_july_demo/inference_examples/1-biplane.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/2024_july_demo/inference_examples/1-cargo_jet.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/2024_july_demo/inference_examples/2-fighter_jet.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/2024_july_demo/inference_examples/3-machine_gun.jpeg',
    ]

    concepts_to_visualize = [
        'biplane',
        'biplane',
        'cargo jet',
        'fighter jet',
        'machine gun'
    ]

    out_dir = '/shared/nas2/blume5/fa23/ecole/july_demo_inference_heatmaps'
    clamp_center = 5

    assert len(test_img_paths) == len(concepts_to_visualize)

    heatmap_writer = HeatmapWriter(clamp_center=clamp_center)

    for ckpt_path in tqdm(ckpt_paths):
        concept_kb = ConceptKB.load(ckpt_path)
        heatmap_writer.set_concept_kb(concept_kb)

        ckpt_dirname = os.path.split(os.path.dirname(ckpt_path))[-1]
        ckpt_basename = os.path.basename(ckpt_path).split('.')[0]
        ckpt_out_dir = os.path.join(out_dir, os.path.join(ckpt_dirname, ckpt_basename))
        logger.info(f'Writing checkpoint results to {ckpt_out_dir}')

        for img_path, concept_name in zip(test_img_paths, concepts_to_visualize):
            img_out_dir = os.path.join(ckpt_out_dir, os.path.basename(img_path).split('.')[0])
            heatmap_writer.visualize_heatmaps(img_path, concept_name, img_out_dir)

    print('Done')