import torch
from PIL.Image import Image
import logging
from image_processing import Segmenter, Localizer, bbox_from_mask
from feature_extraction import Sam, GLIPDemo
from utils import ArticleDeterminer
from torchvision.transforms.functional import pil_to_tensor

logger = logging.getLogger(__name__)

def build_localizer_and_segmenter(sam: Sam, desco: GLIPDemo):
    return LocalizerAndSegmenter(Localizer(sam, desco), Segmenter(sam))

class LocalizerAndSegmenter:
    def __init__(self, localizer: Localizer, segmenter: Segmenter):
        self.localizer = localizer
        self.segmenter = segmenter

        self.article_det = ArticleDeterminer()

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