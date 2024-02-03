# %%
'''
    Functions to localize the object region in an image.
'''
import numpy as np
import torch
from rembg import remove, new_session
from PIL.Image import Image
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from segment_anything import SamPredictor
from typing import Union
import logging

logger = logging.getLogger(__file__)

def bbox_from_mask(masks: Union[torch.Tensor, np.ndarray], use_dim_order: bool = False) -> Union[torch.Tensor, np.ndarray]:
    '''
        Given a set of masks, return the bounding box coordinates and plot the bounding box with width bbox_width.

        ### Arguments:
            masks: (n,h,w)
            use_dim_order (Optional[bool]): Use the order of the dimensions of the masks when returning points (i.e.
                height dim, width dim, height dim, width dim). This is as opposed to XYXY, which corresponds to
                (width dim, height dim, width dim, height dim).

        ### Returns:
            (n,4) bounding box tensor or ndarray, depending on input type. First two coordinates specify upper left
            corner, while next two coordinates specify bottom right corner. Return format determined by value of use_dim_order.

        ### Note
        XYXY order can be visualized with:

            ```python
            import matplotlib.pyplot as plt
            from torchvision.utils import draw_bounding_boxes

            img = torch.zeros(3, 300, 500).int()
            boxes = torch.tensor([[50, 100, 300, 275]])

            plt.imshow(draw_bounding_boxes(img, boxes=boxes).permute(1, 2, 0))
            ```

        NOTE: This is also implemented in torchvision.ops.masks_to_boxes, but their implementation is not vectorized wrt
        number of masks.
    '''
    is_np = isinstance(masks, np.ndarray)
    if is_np:
        masks = torch.from_numpy(masks)

    # Convert masks to boolean and zero pad the masks to have regions on image edges
    # use the edges as boundaries
    masks = masks.bool()

    # Find pixels bounding mask
    top_inds = (
        masks.any(dim=2) # Which rows have nonzero values; (n,h)
             .int() # Argmax not implemented for bool, so convert back to int
             .argmax(dim=1) # Get the first row with nonzero values; (n,)
    )

    left_inds = masks.any(dim=1).int().argmax(dim=1) # Get the first column with nonzero values; (n,)
    bottom_inds = masks.shape[1] - 1 - masks.flip(dims=(1,)).any(dim=2).int().argmax(dim=1) # Reverse rows to get last row with nonzero vals; (n,)
    right_inds = masks.shape[2] - 1 - ( # Since reversing, subtract from total width
        masks.flip(dims=(2,)).any(dim=1) # Reverse columns to get last column with nonzero vals; (n,h)
             .int() # Argmax not implemented for bool, so convert back to int
             .argmax(dim=1) # Get the first column with nonzero values; (n,)
    )

    if use_dim_order: # Specify UL, BR by order of image dimensions: (h, w, h, w)
        upper_lefts = torch.cat([top_inds[:, None], left_inds[:, None]], dim=1) # (n,2)
        bottom_rights = torch.cat([bottom_inds[:, None], right_inds[:, None]], dim=1) # (n,2)

    else: # Specify UL, BR by order of XYXY: (w, h, w, h)
        upper_lefts = torch.cat([left_inds[:, None], top_inds[:, None]], dim=1) # (n,2)
        bottom_rights = torch.cat([right_inds[:, None], bottom_inds[:, None]], dim=1) # (n,2)

    boxes = torch.cat([upper_lefts, bottom_rights], dim=1) # (n,4)

    if is_np:
        boxes = boxes.numpy()

    return boxes

class Localizer:
    def __init__(self, sam: SamPredictor, desco: GLIPDemo, rembg_model_name: str = 'sam_prompt'):
        self.rembg_session = new_session(model_name=rembg_model_name)
        self.desco = desco
        self.sam = sam

    def rembg_ground(self, img: Image) -> torch.IntTensor:
        '''
            Returns (x1, y1, x2, y2) bounding box IntTensor.
        '''
        mask = self.rembg_mask(img) # (h,w)

        logger.info('Obtaining bounding box from rembg mask')
        bbox = bbox_from_mask(mask[None, ...])[0] # (4,)

        # Attempt to widen bounding box since rembg segments the foreground without a bbox
        for i, coord in enumerate(bbox):
            if i < 2 and coord > 0:
                bbox[i] -= 1
            elif (i == 2 and coord + 1 < mask.shape[1]) or (i == 3 and coord + 1 < mask.shape[0]):
                bbox[i] += 1

        return bbox

    def rembg_mask(self, img: Image) -> torch.BoolTensor:
        '''
            Returns (h,w) boolean array.
        '''
        logger.info('Obtaining rembg mask')
        mask = remove(img, session=self.rembg_session, post_process_mask=True, only_mask=True)
        mask = torch.from_numpy(np.array(mask)).bool()

        return mask

    def desco_ground(self, img: Image, caption: str, token_to_ground: str, conf_thresh: float = .4) -> torch.IntTensor:
        '''
            Based on the contents of predictor_glip.py.GLIPDemo.run_on_web_image.
            This function signature must be changed if we wish to use the GLIPDemo from predictor_FIBER.py

            XXX The load method of run_demo.py maps the image to BGR, but this seems to conflict with
            GLIPDemo.build_transform's to_bgr_transform. We therefore ignore run_demo.py and don't load in BGR.

            Returns (x1, y1, x2, y2) bounding box IntTensor.
        '''
        # Get bounding box with DesCo
        logger.info('Obtaining bounding box with DesCo')

        caption = caption.lower()
        token_to_ground = token_to_ground.lower()

        predictions = self.desco.compute_prediction(np.array(img), caption, specified_tokens=[token_to_ground])
        top_predictions: BoxList = self.desco._post_process(predictions, conf_thresh)

        bboxes = top_predictions.bbox.cpu() # (n_boxes, 4)

        if len(bboxes) == 0:
            log_str = f'Failed to ground token with conf_thresh={conf_thresh}'
            logger.warning(log_str)
            raise RuntimeError(log_str)

        logger.info(f'Detected {len(bboxes)} bounding boxes; taking the first one')

        return bboxes[0]

    def desco_mask(self, img: Image, caption: str, token_to_ground: str, conf_thresh: float = .4) -> torch.BoolTensor:
        '''
            Returns (h,w) boolean array.
        '''
        box = self.desco_ground(img, caption, token_to_ground, conf_thresh).numpy() # SAM takes numpy bbox

        logger.info('Obtaining segmentation mask with SAM')
        self.sam.set_image(np.array(img))
        masks, scores, logits = self.sam.predict(box=box, multimask_output=True)

        mask = torch.from_numpy(masks[np.argmax(scores)]).bool()

        return mask

    def localize(self, img: Image, caption='', token_to_ground: str = '', conf_thresh: float = .4) -> torch.IntTensor:
        '''
            Returns boolean array of shape (h, w)
        '''
        img = img.convert('RGB')

        if caption:
            assert token_to_ground and token_to_ground in caption
            logger.info(f'Localizing with DesCo to ground "{token_to_ground}" with caption "{caption}"')
            bbox = self.desco_ground(img, caption, token_to_ground, conf_thresh)

        else: # rembg
            logger.info('Localizing with rembg')
            bbox = self.rembg_ground(img)

        return bbox

# %%
if __name__ == '__main__':
    # Imports
    import PIL
    from vis_utils import show, image_from_masks
    from torchvision.utils import draw_bounding_boxes
    from torchvision.transforms.functional import pil_to_tensor
    from models import build_sam_predictor, build_desco
    import coloredlogs

    coloredlogs.install(level=logging.INFO, logger=logger)

    # %% Construct localizer
    sam = build_sam_predictor()
    desco = build_desco()
    localizer = Localizer(sam, desco)

    # %% Test
    img_path = '/shared/nas2/blume5/fa23/ecole/data/inaturalist2021/images/train_mini/00029_Animalia_Arthropoda_Arachnida_Araneae_Araneidae_Gasteracantha_kuhli/4f84deea-bca0-4f4b-ac94-828828e089da.jpg'
    caption = token = 'animal'

    img = PIL.Image.open(img_path).convert('RGB')
    img_tensor = pil_to_tensor(img)

    desco_bbox = localizer.desco_ground(img, caption, token)
    desco_mask = localizer.desco_mask(img, caption, token)

    show([
        draw_bounding_boxes(img_tensor, boxes=desco_bbox.unsqueeze(0), colors='red'),
        image_from_masks(desco_mask.unsqueeze(0), superimpose_on_image=img_tensor)
    ], title='DesCo')

    rembg_bbox = localizer.rembg_ground(img)
    rembg_mask = localizer.rembg_mask(img)

    show([
        draw_bounding_boxes(img_tensor, boxes=rembg_bbox.unsqueeze(0), colors='red'),
        image_from_masks(rembg_mask.unsqueeze(0), superimpose_on_image=img_tensor)
    ], title='Rembg')
# %%
