# %%
'''
    Functions to split a localized region into parts.

    1. Generate crop around region's bounding box
    2. Potentially remove background from object, setting background to zero
    3. Use fine-grained SAM to segment object into parts
'''
import torch
from segment_anything.modeling import Sam
from feature_extraction import build_sam_amg
from PIL.Image import Image
from rembg import remove, new_session
from torchvision.transforms.functional import crop, to_pil_image, pil_to_tensor
from torchvision.ops import box_convert
import numpy as np
from image_processing.localize import bbox_from_mask
import logging
logger = logging.getLogger(__name__)

class Segmenter:
    def __init__(self, sam: Sam, rembg_model_name: str = 'isnet-general-use'):
        self.sam_amg = build_sam_amg(sam, part_based=True)
        self.rembg_session = new_session(model_name=rembg_model_name)

    def remove_background(self, image: Image)-> Image:
        '''
            Removes background from image using rembg.

            Arguments:
                image (PIL.Image.Image): Image to remove background from
        '''
        return remove(image, session=self.rembg_session, post_process_mask=True)

    def foreground_mask(self, image: Image)-> torch.BoolTensor:
        '''
            Determines the foreground mask of the image using rembg.

            Arguments:
                image (PIL.Image.Image): Image to extract foreground mask from
        '''
        mask: Image = remove(image, session=self.rembg_session, post_process_mask=True, only_mask=True) # bw image
        mask = np.array(mask) > 0

        return torch.tensor(mask)

    def crop(self, image: Image, bbox: torch.IntTensor, remove_background=False) -> Image:
        '''
            Crops the region specified by bbox from the image.

            Arguments:
                image (PIL.Image.Image): Image to crop
                bbox (torch.Tensor): Bounding box of object to crop in XYXY
                remove_background (bool): Whether to remove background from object after cropping
        '''
        assert isinstance(bbox, torch.IntTensor), 'bbox tensor must use ints'
        left, top, width, height = box_convert(bbox[None, ...], 'xyxy', 'xywh')[0]

        cropped = crop(image, top.item(), left.item(), height.item(), width.item())

        if remove_background:
            cropped = self.remove_background(cropped).convert('RGB')

        return cropped

    def crops_from_masks(self, image: Image, masks: torch.BoolTensor, remove_background=False, only_mask=False) -> list[Image]:
        '''
            Crops the regions specified by masks from the image.
            Arguments:
                image (PIL.Image.Image): Image to crop
                masks (torch.BoolTensor): Boolean mask (n_masks, h, w ) of segmented parts.
                remove_background (bool): Whether to remove background from object after cropping with rembg.
                only_mask (bool): Whether to zero out non-masked regions in the cropped image.
        '''
        if len(masks) == 0:
            return []

        bboxes = bbox_from_mask(masks)

        crops = []
        image_t = pil_to_tensor(image)
        for bbox, mask in zip(bboxes, masks):
            if only_mask: # Set non-masked regions to zero
                image = to_pil_image(image_t * mask)

            crops.append(self.crop(image, bbox, remove_background))

        return crops

    def segment(
        self,
        image: Image,
        bbox: torch.Tensor,
        remove_background=False,
        pixel_threshold=100,
        segment_before_crop=False
    ) -> torch.BoolTensor:
        '''
            Segments the region specified by bbox into parts.

            Arguments:
                image (PIL.Image.Image): Image to segment
                bbox (torch.Tensor): Bounding box of object to segment in XYXY
                remove_background (bool): Whether to remove background from object after cropping
                pixel_threshold (int): Minimum number of pixels for a mask to be considered a part
                segment_before_crop (bool): Whether to segment the entire image before cropping results, instead
                    of cropping the image first and then segmenting the cropped image.

            Returns:
                (torch.BoolTensor): Boolean mask (n_masks, h, w ) of segmented parts.
        '''
        if remove_background or not segment_before_crop:
            cropped_image = self.crop(image, bbox, remove_background)

        try:
            masks: list[dict] = self.sam_amg.generate(np.array(image if segment_before_crop else cropped_image))
        except ValueError: # Failed to segment due to crop being too small
            logger.warning('Failed to segment image. Returning full mask.')
            shape = image.size[::-1] if segment_before_crop else cropped_image.size[::-1]
            return torch.full((1, *shape), True, dtype=torch.bool)

        masks = sorted(masks, key=lambda mask_d: mask_d['area'], reverse=True) # Sort by area
        masks = torch.stack([torch.from_numpy(mask_d['segmentation']) for mask_d in masks]).bool() # (n_masks, h_crop, w_crop) or (n_masks, h, w)

        x1, y1, x2, y2 = bbox

        if segment_before_crop:
            # Create mask of ones specified by bbox and zero elsewhere
            bbox_mask = torch.zeros(image.size[1], image.size[0], dtype=torch.bool)
            bbox_mask[y1:y2, x1:x2] = True

            if remove_background: # Remove background based on bbox crop
                bbox_mask[y1:y2, x1:x2] = bbox_mask[y1:y2, x1:x2] & self.foreground_mask(cropped_image)

            masks = torch.stack([m & bbox_mask for m in masks]) # Restrict mask to bbox

        else:
            if remove_background:
                masks = masks & self.foreground_mask(cropped_image)

            # Convert back to original image size
            full_masks = torch.zeros(len(masks), image.size[1], image.size[0], dtype=torch.bool)
            for i, crop_mask in enumerate(masks):
                full_masks[i, y1:y2, x1:x2] = crop_mask

            masks = full_masks

        # Filter out small masks, including those which may have been removed by background removal
        orig_num_masks = len(masks)
        masks = [m for m in masks if m.sum() > pixel_threshold]
        logger.info(f'Filtered out {orig_num_masks - len(masks)} / {orig_num_masks} masks')

        if len(masks) == 0:
            logger.warning('No masks remaining after filtering. Consider lowering pixel_threshold.')
            masks = torch.tensor([])

        else:
            masks = torch.stack([m for m in masks if m.sum() > pixel_threshold])

        return masks

# %%
if __name__ == '__main__':
    import PIL
    from vis_utils import image_from_masks, show
    from feature_extraction import build_sam, build_desco
    from mo9_demo.src.image_processing.localize import Localizer
    from torchvision.utils import draw_bounding_boxes
    from torchvision.transforms.functional import pil_to_tensor

    img_path = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/dog.png'
    device = 'cuda'

    img = PIL.Image.open(img_path).convert('RGB')
    sam = build_sam(device=device)
    segmenter = Segmenter(sam)

    desco = build_desco(device=device)
    localizer = Localizer(sam, desco)

    # %%
    bbox = localizer.localize(img, caption='animal', token_to_ground='animal')
    show(
        draw_bounding_boxes(pil_to_tensor(img), bbox.unsqueeze(0), colors='red', width=4),
        title='Grounded Bounding Box'
    )

    show(segmenter.crop(img, bbox, remove_background=False), title='Cropped Image')
    show(segmenter.crop(img, bbox, remove_background=True), title='Cropped Image with Background Removed')

    show(
        image_from_masks(segmenter.segment(img, bbox, remove_background=False)),
        title='Segmented Parts'
    )

    show(
        image_from_masks(segmenter.segment(img, bbox, remove_background=True)),
        title='Segmented Parts with Background Removed'
    )

    show(
        segmenter.crops_from_masks(img, segmenter.segment(img, bbox, remove_background=True), remove_background=True),
        title='Cropped Segmented Parts'
    )
# %%
