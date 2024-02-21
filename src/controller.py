# %%
if __name__ == '__main__': # TODO Delete me after debugging
    import sys
    sys.path.append('/shared/nas2/blume5/fa23/ecole/src/mo9_demo/src')

from score import AttributeScorer
from model.concept import ConceptKB, Concept
from PIL.Image import Image
import logging, coloredlogs
from feature_extraction import build_sam, build_desco, FeatureExtractor
from image_processing import LocalizerAndSegmenter, build_localizer_and_segmenter
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from kb_ops.train import ConceptKBTrainer
from llm import LLMClient, retrieve_parts, retrieve_attributes
from score import AttributeScorer
from feature_extraction import CLIPAttributePredictor
from utils import to_device
from vis_utils import plot_predicted_classes

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

class Controller:
    def __init__(
        self,
        loc_and_seg: LocalizerAndSegmenter,
        concept_kb: ConceptKB,
        feature_extractor: FeatureExtractor,
        zs_predictor: CLIPAttributePredictor = None
    ):
        self.concepts = concept_kb
        self.trainer = ConceptKBTrainer(concept_kb, feature_extractor, loc_and_seg)

        self.loc_and_seg = loc_and_seg
        self.llm_client = LLMClient()
        self.attr_scorer = AttributeScorer(zs_predictor)

        self.cached_predictions = []

    ##############
    # Prediction #
    ##############
    def clear_cached_predictions(self):
        self.cached_predictions = []

    def predict_concept(self, image: Image, unk_threshold: float = .1):
        prediction = self.trainer.predict(image_data=image, unk_threshold=unk_threshold)
        self.cached_predictions.append(prediction)

        img = plot_predicted_classes(prediction, threshold=unk_threshold, return_img=True)
        predicted_label = prediction['predicted_label']
        predicted_label = predicted_label if predicted_label != self.trainer.UNK_LABEL else 'unknown'

        return {
            'predicted_label': predicted_label,
            'plot': img
        }

    def localize_and_segment(
        self,
        image: Image,
        concept_name: str = '',
        concept_parts: list[str] = [],
        remove_background: bool = True,
        return_crops: bool = True
    ):
        return self.loc_and_seg.localize_and_segment(
            image=image,
            concept_name=concept_name,
            concept_parts=concept_parts,
            remove_background=remove_background,
            return_crops=return_crops
        )

    def predict_from_zs_attributes(
        self,
        image: Image,
        concept_name: str = '',
        concept_parts: list[str] = [],
        remove_background: bool = True,
        zs_attrs: list[str] = []
    ) -> dict:
        segmentations = self.loc_and_seg.localize_and_segment(
            image=image,
            concept_name=concept_name,
            concept_parts=concept_parts,
            remove_background=remove_background,
            return_crops=True
        )

        region_crops = segmentations['part_crops']
        if region_crops == []:
            region_crops = [image]

        # Compute zero-shot scores
        part_zs_scores = to_device(self.zs_attr_match_score(region_crops, zs_attrs), 'cpu')
        full_zs_scores = to_device(self.zs_attr_match_score([image], zs_attrs), 'cpu')

        ret_dict = {
            'segmentations': segmentations,
            'scores': {
                'part_zs_scores': part_zs_scores,
                'full_zs_scores': full_zs_scores
            }
        }

        return ret_dict

    ###########
    # Scoring #
    ###########
    def zs_attr_match_score(self, regions: list[Image], zs_attrs: list[str]):
        results = self.attr_scorer.score_regions(regions, zs_attrs)

        raw_scores = results['zs_scores_per_region_raw'] # (n_regions, n_texts)
        weighted_scores = results['zs_scores_per_region_weighted'] # (n_regions, n_texts)

        attr_probs_per_region = weighted_scores.permute(1, 0).softmax(dim=1) # (n_texts, n_regions)
        match_score = weighted_scores.sum().item()

        ret_dict = {
            'raw_scores': raw_scores,
            'weighted_scores': weighted_scores,
            'attr_probs_per_region': attr_probs_per_region,
            'match_score': match_score
        }

        return ret_dict

    ###################
    # Concept Removal #
    ###################
    def clear_concepts(self):
        pass

    def remove_concept(self, concept_name: str):
        pass

    ####################
    # Concept Addition #
    ####################
    def add_concept(self, concept: Concept):
        # Get zero shot attributes (query LLM)

        # Determine if it has any obvious parent or child concepts

        # Get likely learned attributes

        # Add concept

        pass

    def _get_zs_attributes(self, concept_name: str):
        pass

    ########################
    # Concept Modification #
    ########################
    def add_zs_attribute(self, concept_name: str, zs_attr_name: str, weight: float):
        pass

    def remove_zs_attribute(self, concept_name: str, zs_attr_name: str):
        pass

    def add_learned_attribute(self, concept_name: str, learned_attr_name: str, weight: float):
        pass

    def remove_learned_attribute(self, concept_name: str, learned_attr_name: str):
        pass

    ##################
    # Interpretation #
    ##################
    def compare_concepts(self, concept1_name: str, concept2_name: str):
        pass

# %%
if __name__ == '__main__':
    import PIL
    import os
    from torchvision.utils import draw_bounding_boxes
    from vis_utils import image_from_masks

    sam = build_sam()
    desco = build_desco()
    llm_client = LLMClient()
    loc_and_seg = build_localizer_and_segmenter(sam, desco)

    # %% Path
    in_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/graduate_descent'
    out_dir = '/shared/nas2/blume5/fa23/ecole/results/2_11_24-graduate_descent'

    os.makedirs(out_dir, exist_ok=True)

    # %%
    def run_segmentation(concept_name, concept_parts, file_fmt, save_crops=False):
        try:
            result = loc_and_seg.localize_and_segment(img, concept_name=concept_name, concept_parts=concept_parts)

        except RuntimeError as e:
            logger.error(f'Error occurred during segmentation: {e}')
            return

        # Save localized region
        bbox_img = draw_bounding_boxes(pil_to_tensor(img), result['localized_bbox'].unsqueeze(0), colors='red', width=8)
        to_pil_image(bbox_img).save(os.path.join(out_dir, file_fmt.format('localized')))

        # Save part bboxes, if available
        if 'localized_part_bboxes' in result:
            part_bbox_img = draw_bounding_boxes(pil_to_tensor(img), result['localized_part_bboxes'], colors='green', width=8)
            to_pil_image(part_bbox_img).save(os.path.join(out_dir, file_fmt.format('part_bboxes')))

        # Save segmented region
        mask_img = image_from_masks(result['part_masks'], superimpose_on_image=pil_to_tensor(img))
        to_pil_image(mask_img).save(os.path.join(out_dir, file_fmt.format('segmented')))

        # Save crops
        if save_crops:
            for i, crop in enumerate(result['part_crops']):
                crop.save(os.path.join(out_dir, file_fmt.format(f'part_{i}')))

        return result

    # %%
    for basename in os.listdir(in_dir):
        if not (basename.endswith('.jpg') or basename.endswith('.png')):
            continue

        logger.info(f'Processing {basename}')
        input_path = os.path.join(in_dir, basename)
        img = PIL.Image.open(input_path).convert('RGB')

        # Extract file name for saving and object name for prompting
        fname = os.path.splitext(basename)[0]
        obj_name = fname.split('_')[0]

        # No concept name, no concept parts
        result = run_segmentation('', [], f'{fname}-no_name-no_parts-{{}}.jpg')

        # Concept name, no concept parts
        result = run_segmentation(obj_name, [], f'{fname}-name-no_parts-{{}}.jpg')

        # Concept name, concept parts
        retrieved_parts = retrieve_parts(obj_name, llm_client)
        result = run_segmentation(obj_name, retrieved_parts, f'{fname}-name-parts-{{}}.jpg')
    # %%
    #
