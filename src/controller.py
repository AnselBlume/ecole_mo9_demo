# %%
if __name__ == '__main__': # TODO Delete me after debugging
    import sys
    sys.path.append('/shared/nas2/blume5/fa23/ecole/src/mo9_demo/src')

from score import AttributeScorer
from model.concept import ConceptKB, Concept
from image_processing.segment import Segmenter
from image_processing.localize import Localizer, bbox_from_mask
from PIL.Image import Image
import logging, coloredlogs
from feature_extraction import build_sam, build_desco, Sam, GLIPDemo
import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from llm import LLMClient, retrieve_parts, retrieve_attributes
from score import AttributeScorer
from feature_extraction import CLIPAttributePredictor
import torch.nn.functional as F
from utils import to_device, ArticleDeterminer

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

class Controller:
    def __init__(self, sam: Sam, desco: GLIPDemo, concept_db: ConceptKB, zs_predictor: CLIPAttributePredictor = None):
        self.concepts = concept_db

        self.segmenter = Segmenter(sam)
        self.localizer = Localizer(sam, desco)
        self.llm_client = LLMClient()
        self.attr_scorer = AttributeScorer(zs_predictor)

        self.article_det = ArticleDeterminer()

    ################
    # Segmentation #
    ################
    def predict_image(self, image: Image):
        pass

    @torch.inference_mode()
    def localize_and_segment(
        self,
        image: Image,
        concept_name: str = '',
        concept_parts: list[str] = [],
        remove_background: bool = True,
        return_crops: bool = True
    ) -> dict:
        '''
            Localizes and segments the concept in the image in to parts.

            Arguments:
                image (PIL.Image.Image): Image to localize and segment
                concept_name (str): Name of concept to localize. If not provided, uses rembg to perform foreground segmentation
                concept_parts (list[str]): List of part names to localize. If not provided, uses SAM to perform part segmentation
                remove_background (bool): Whether to remove the background from the localized concept.
                return_crops (bool): If true, returns images of the cropped parts (possibly with background removed) under the key 'part_crops'.

            Returns:
                dict: Dictionary containing the following keys:
                    'part_masks' (torch.BoolTensor): Boolean array of shape (n_detections, h, w) representing the segmentation masks of the parts
                    'localized_bbox' (torch.IntTensor): Bounding box of the localized concept in XYXY format with shape (4,).
                    'localized_part_bboxes' (torch.IntTensor): Tensor of shape (n_part_detections, 4) of bounding boxes of the localized parts in XYXY format.
                    'part_crops' (list[PIL.Image.Image]): List of cropped part images, if return_crops is True
        '''
        # Localize the concept
        caption = self._get_parts_caption(concept_name, concept_parts) if concept_parts else concept_name
        bboxes = self.localizer.localize(image, caption=caption, tokens_to_ground=[concept_name])

        if len(bboxes) == 0: # Fall back to rembg if DesCo fails
            if concept_name:
                logger.warning('Failed to ground concept with caption; retrying with rembg')
                bboxes = self.localizer.localize(image)

            if len(bboxes) == 0:
                log_str = 'Failed to localize concept with rembg'
                logger.error(log_str)
                raise RuntimeError(log_str)

        # Segment the concept parts
        if concept_parts:
        # if concept_name in self.concepts: # Use DesCo if we can retrieve the concept parts
            logger.info(f'Localizing concept parts {concept_parts} with DesCo')
            # concept = self.concepts.get_concept(concept_name)
            # component_parts = list(concept.component_concepts.keys())
            # TODO consider setting areas not in the bbox to zero instead of cropping the image to maintain scale
            cropped_image = self.segmenter.crop(image, bboxes[0], remove_background=remove_background) # Crop around localized concept
            part_masks = self.localizer.desco_mask(cropped_image, caption=caption, tokens_to_ground=concept_parts) # (n_detections, h, w)

            logger.info(f'Obtained {len(part_masks)} part masks with DesCo')

            if len(part_masks) == 0:
                raise RuntimeError('Failed to localize concept parts with DesCo')

            # Convert masks into full image size
            full_part_masks = []
            x1, y1, x2, y2 = bboxes[0]

            if remove_background: # Crop had background removed, so just extract mask
                crop_foreground_mask = pil_to_tensor(cropped_image).bool().sum(dim=0).bool()

            else: # Don't remove background
                crop_foreground_mask = torch.ones(cropped_image.size[1], cropped_image.size[0], dtype=torch.bool)

            for i, part_mask in enumerate(part_masks):
                full_part_mask = torch.zeros(image.size[1], image.size[0], dtype=torch.bool)
                full_part_mask[y1:y2, x1:x2] = part_mask & crop_foreground_mask

                if full_part_mask.sum() == 0:
                    logger.warning(f'Part mask {i} is empty after intersecting with foreground; skipping')
                    continue

                full_part_masks.append(full_part_mask)

            part_masks = torch.stack(full_part_masks)

        else: # Non part-based segmentation of localized concept
            logger.info('Performing part segmentation with SAM')
            part_masks = self.segmenter.segment(image, bboxes[0], remove_background=remove_background)

        # Construct return dictionary
        ret_dict = {
            'part_masks': part_masks,
            'localized_bbox': bboxes[0],
        }

        if return_crops:
            part_crops = self.segmenter.crops_from_masks(image, part_masks, only_mask=remove_background)
            logger.info(f'Generated {len(part_crops)} part crops from part masks')

            # Ignore part crops with a zero-dimension (caused by part mask being one-dimensional, e.g. a line)
            filtered_crop_parts = []
            for i, crop in enumerate(part_crops):
                if crop.size[0] == 0 or crop.size[1] == 0:
                    logger.warning(f'Part crop {i} has a zero-dimension; adding None to part_crops instead of crop')
                    filtered_crop_parts.append(None)

                else:
                    filtered_crop_parts.append(crop)

            ret_dict['part_crops'] = filtered_crop_parts

        if concept_parts:
            ret_dict['localized_part_bboxes'] = bbox_from_mask(part_masks)

        ret_dict = {k : ret_dict[k] for k in sorted(ret_dict.keys())}

        return ret_dict

    def _get_parts_caption(self, concept_name: str, component_parts: list[str]):
        '''
            dog, head, whiskers, tail --> a dog with a head, whiskers, and a tail
        '''
        prompt = f'{self.article_det.determine(concept_name)}{concept_name} '
        for i, component_part in enumerate(component_parts):
            if i == 0:
                prompt += 'with '
            elif i == len(component_parts) - 1:
                prompt += ', and ' if len(component_parts) > 2 else ' and '
            else:
                prompt += ', '

            prompt += f'{self.article_det.determine(component_part)}{component_part}'

        return prompt

    ##############
    # Prediction #
    ##############
    def predict(self,
        image: Image,
        concept_name: str = '',
        concept_parts: list[str] = [],
        remove_background: bool = True,
        zs_attrs: list[str] = []
    ) -> dict:
        segmentations = self.localize_and_segment(
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
    controller = Controller(sam, desco, ConceptKB())

    # %% Path
    in_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/graduate_descent'
    out_dir = '/shared/nas2/blume5/fa23/ecole/results/2_11_24-graduate_descent'

    os.makedirs(out_dir, exist_ok=True)

    # %%
    def run_segmentation(concept_name, concept_parts, file_fmt, save_crops=False):
        try:
            result = controller.localize_and_segment(img, concept_name=concept_name, concept_parts=concept_parts)

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
