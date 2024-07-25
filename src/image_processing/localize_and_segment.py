import logging
from dataclasses import dataclass, field
from typing import Union

import torch
from feature_extraction import Sam
from image_processing.localize import Localizer, bbox_from_mask
from image_processing.segment import Segmenter
from model.dataclass_base import DeviceShiftable, DictDataClass
from PIL.Image import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import draw_bounding_boxes
from utils import ArticleDeterminer

logger = logging.getLogger(__name__)

@dataclass
class LocalizeAndSegmentOutput(DictDataClass, DeviceShiftable):
    input_image: Image = field(
        default=None,
        metadata={'description': 'Input image'}
    )

    input_image_path: str = field(
        default=None,
        metadata={'description': 'Path corresponding to input image'}
    )

    input_kwargs: dict = field(
        default=None,
        metadata={'description': 'Input keyword arguments'}
    )

    part_masks: torch.BoolTensor = field(
        default=None,
        metadata={'description': 'Boolean array of shape (n_detections, h, w) representing the segmentation masks of the parts'}
    )

    part_names: list[str] = field(
        default=None,
        metadata={'dsecription': 'List of part names corresponding to part_masks. None if part regions are unnamed (e.g. if segmented without grounding by name).'}
    )

    rle_part_masks: list[dict] = field(
        default=None,
        metadata={'description': 'RLE-encoded masks output by pycocotools.mask.encode; currently unused'}
    )

    localized_bbox: torch.IntTensor = field(
        default=None,
        metadata={'description': 'Bounding box of the localized concept in XYXY format with shape (4,)'}
    )

    object_mask: torch.BoolTensor = field(
        default=None,
        metadata={'description': 'Mask of the localized concept corresponding to the localized bbox with shape (h, w)'}
    )

    part_crops: list[Image] = field(
        default=None,
        metadata={'description': 'List of cropped part images, if return_crops is True'}
    )

    localized_part_bboxes: torch.IntTensor = field(
        default=None,
        metadata={'description': 'Tensor of shape (n_part_detections, 4) of bboxes of localized parts in XYXY format, if concept_parts is provided.'}
    )

@dataclass
class LocalizerAndSegmenterConfig:
    do_localize: bool = True
    remove_background: bool = True
    return_crops: bool = True
    use_bbox_for_crops: bool = False

    def __post_init__(self):
        if not self.do_localize and self.remove_background:
            logger.warning('remove_background is true while do_localize is false; setting remove_background to false')
            self.remove_background = False

class LocalizerAndSegmenter:
    def __init__(
        self,
        localizer: Localizer,
        segmenter: Segmenter,
        config: LocalizerAndSegmenterConfig = LocalizerAndSegmenterConfig()
    ):
        self.localizer = localizer
        self.segmenter = segmenter
        self.config = config

        self.article_det = ArticleDeterminer()

    def to(self, device: Union[str, torch.device]):
        self.localizer.to(device)
        self.segmenter.to(device)

        return self

    @torch.inference_mode()
    def localize_and_segment(
        self,
        image: Image,
        concept_name: str = '',
        concept_parts: list[str] = [],
        do_localize: bool = None,
        remove_background: bool = None,
        return_crops: bool = None,
        use_bbox_for_crops: bool = None
    ) -> LocalizeAndSegmentOutput:
        '''
            Localizes and segments the concept in the image in to parts.

            Arguments:
                image (PIL.Image.Image): Image to localize and segment
                concept_name (str): Name of concept to localize. If not provided, uses rembg to perform foreground segmentation
                concept_parts (list[str]): List of part names to localize. If not provided, uses SAM to perform part segmentation
                do_localize (bool): Whether to localize the concept in the image or use the whole image. If false, disables background removal.
                remove_background (bool): Whether to remove the background from the localized concept.
                return_crops (bool): If true, returns images of the cropped parts (possibly with background removed) under the key 'part_crops'.
                use_bbox_for_crops (bool): If true, draws bounding boxes around the parts instead of using the part masks to crop the parts.

            Returns:
                LocalizeAndSegmentOutput
                    'part_masks' (torch.BoolTensor): Boolean array of shape (n_detections, h, w) representing the segmentation masks of the parts
                    'localized_bbox' (torch.IntTensor): Bounding box of the localized concept in XYXY format with shape (4,).
                    'localized_part_bboxes' (torch.IntTensor): Tensor of shape (n_part_detections, 4) of bounding boxes of the localized parts in XYXY format.
                    'part_crops' (list[PIL.Image.Image]): List of cropped part images, if return_crops is True
                    'localized_part_bboxes' (torch.IntTensor): Tensor of shape (n_part_detections, 4) of bounding boxes of the localized parts in XYXY format,
                        if concept_parts is provided.
        '''
        do_localize = do_localize if do_localize is not None else self.config.do_localize
        remove_background = remove_background if remove_background is not None else self.config.remove_background
        return_crops = return_crops if return_crops is not None else self.config.return_crops
        use_bbox_for_crops = use_bbox_for_crops if use_bbox_for_crops is not None else self.config.use_bbox_for_crops

        # Localize the concept
        caption = self._get_parts_caption(concept_name, concept_parts) if concept_parts else concept_name

        if do_localize:
            bboxes, object_masks = self.localizer.localize(image, caption=caption, tokens_to_ground=[concept_name], return_object_masks=True)
        else:
            bboxes = torch.tensor([0, 0, image.size[0], image.size[1]], dtype=torch.int32)[None,...] # (x1, y1, x2, y2) = (0, 0, w, h)
            object_masks = torch.ones(1, image.size[1], image.size[0], dtype=torch.bool) # (1, h, w)

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
        output = LocalizeAndSegmentOutput(
            part_masks=part_masks,
            localized_bbox=bboxes[0],
            object_mask=object_masks[0]
        )

        if return_crops:
            if use_bbox_for_crops:
                image_t = pil_to_tensor(image)
                bboxes = bbox_from_mask(part_masks)

                part_crops = [
                    to_pil_image(draw_bounding_boxes(image_t, bbox[None,...], width=3, colors='red'))
                    for bbox in bboxes
                ]

            else:
                part_crops = self.segmenter.crops_from_masks(image, part_masks, only_mask=remove_background)

            logger.info(f'Generated {len(part_crops)} part crops from part masks')

            # Ignore part masks where the crop has a zero-dimension (caused by part mask being one-dimensional, e.g. a line)
            filtered_part_masks = []
            filtered_crop_parts = []
            for i, crop in enumerate(part_crops):
                if crop.size[0] == 0 or crop.size[1] == 0:
                    logger.warning(f'Part crop {i} has a zero-dimension; filtering out part mask and crop')
                    continue

                else:
                    filtered_part_masks.append(part_masks[i])
                    filtered_crop_parts.append(crop)

            output.part_masks = torch.stack(filtered_part_masks) if filtered_part_masks else torch.tensor([])
            output.part_crops = filtered_crop_parts

        if concept_parts:
            output.localized_part_bboxes = bbox_from_mask(part_masks)

        return output

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

def build_localizer_and_segmenter(sam: Sam, desco, config: LocalizerAndSegmenterConfig = LocalizerAndSegmenterConfig()):
    return LocalizerAndSegmenter(Localizer(sam, desco), Segmenter(sam), config=config)