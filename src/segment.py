# %%
'''
    Functions to split a localized region into parts.

    1. Generate crop around region's bounding box
    2. Potentially remove background from object, setting background to zero
    3. Use fine-grained SAM to segment object into parts
'''
import torch
from segment_anything.modeling import Sam
from predictors import build_sam_amg
from PIL.Image import Image
from rembg import remove, new_session
from torchvision.transforms.functional import crop
from torchvision.ops import box_convert
import numpy as np
from localize import bbox_from_mask

class Segmenter:
    def __init__(self, sam: Sam, rembg_model_name: str = 'sam_prompt'):
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

    def crops_from_masks(self, image: Image, masks: torch.BoolTensor, remove_background=False) -> list[Image]:
        bboxes = bbox_from_mask(masks)
        return [self.crop(image, bbox, remove_background) for bbox in bboxes]

    def segment(self, image: Image, bbox: torch.Tensor, remove_background=False) -> torch.BoolTensor:
        '''
            Segments the region specified by bbox into parts.

            Arguments:
                image (PIL.Image.Image): Image to segment
                bbox (torch.Tensor): Bounding box of object to segment in XYXY
                remove_background (bool): Whether to remove background from object after cropping

            Returns:
                (torch.BoolTensor): Boolean mask (n_masks, h, w ) of segmented parts.
        '''
        image = self.crop(image, bbox, remove_background)

        masks: list[dict] = self.sam_amg.generate(np.array(image))
        masks = torch.stack([torch.from_numpy(mask_d['segmentation']) for mask_d in masks]).bool()

        if remove_background: # Remove masks segmenting the background
            masks = masks & self.foreground_mask(image)
            masks = torch.stack([m for m in masks if m.sum() > 0])

        return masks

# %%
if __name__ == '__main__':
    import PIL
    from vis_utils import image_from_masks, show
    from predictors import build_sam, build_desco
    from localize import Localizer
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
