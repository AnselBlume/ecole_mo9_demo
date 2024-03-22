# %%
'''
    Functions to localize the object region in an image.
'''
import numpy as np
import torch
from rembg import remove, new_session
from PIL.Image import Image
# from maskrcnn_benchmark.structures.bounding_box import BoxList
# from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from segment_anything.modeling import Sam
from feature_extraction import build_sam_predictor
from typing import Union
import logging

logger = logging.getLogger(__file__)

def bbox_from_mask(masks: Union[torch.Tensor, np.ndarray], use_dim_order: bool = False) -> Union[torch.IntTensor, np.ndarray]:
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

    if len(masks) == 0:
        return torch.zeros(0, 4).int() if not is_np else np.zeros((0, 4), dtype=int)

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

    boxes = torch.cat([upper_lefts, bottom_rights], dim=1).int() # (n,4)

    if is_np:
        boxes = boxes.numpy()

    return boxes

class Localizer:
    def __init__(self, sam: Sam, desco, rembg_model_name: str = 'isnet-general-use'):
        self.rembg_session = new_session(model_name=rembg_model_name)
        self.desco = desco
        self.sam = build_sam_predictor(model=sam)

    def rembg_ground(self, img: Image) -> torch.IntTensor:
        '''
            Returns tensor of shape (1, 4), where the last dim specifies (x1, y1, x2, y2),
            the bounding box IntTensor.
        '''
        mask = self.rembg_mask(img) # (1, h,w)

        logger.info('Obtaining bounding box from rembg mask')
        bbox = bbox_from_mask(mask)[0] # (4,)

        # Attempt to widen bounding box since rembg segments the foreground without a bbox
        for i, coord in enumerate(bbox):
            if i < 2 and coord > 0:
                bbox[i] -= 1
            elif (i == 2 and coord + 1 < mask.shape[1]) or (i == 3 and coord + 1 < mask.shape[0]):
                bbox[i] += 1

        return bbox.unsqueeze(0)

    def rembg_mask(self, img: Image) -> torch.BoolTensor:
        '''
            Returns (1, h, w) boolean array.
        '''
        logger.info('Obtaining rembg mask')
        mask = remove(img, session=self.rembg_session, post_process_mask=True, only_mask=True)
        mask = torch.from_numpy(np.array(mask)).bool()

        return mask.unsqueeze(0)

    def desco_ground(self, img: Image, caption: str, tokens_to_ground: Union[str,list[str]], conf_thresh: float = .4) -> torch.IntTensor:
        '''
            Based on the contents of predictor_glip.py.GLIPDemo.run_on_web_image.
            This function signature must be changed if we wish to use the GLIPDemo from predictor_FIBER.py

            XXX The load method of run_demo.py maps the image to BGR, but this seems to conflict with
            GLIPDemo.build_transform's to_bgr_transform. We therefore ignore run_demo.py and don't load in BGR.

            Returns a tensor of shape (n_detected, 4) where the last dimension specifies (x1, y1, x2, y2), the
            bounding box IntTensor. If no bounding boxes detected, returns a tensor of shape (0, 4).
        '''
        # Get bounding box with DesCo
        logger.info('Obtaining bounding box with DesCo')

        caption = caption.lower()

        if isinstance(tokens_to_ground, str):
            tokens_to_ground = [tokens_to_ground]

        tokens_to_ground = [token.lower() for token in tokens_to_ground]

        predictions = self.desco.compute_prediction(np.array(img), caption, specified_tokens=tokens_to_ground)
        top_predictions: BoxList = self.desco._post_process(predictions, conf_thresh)

        bboxes = top_predictions.bbox.cpu() # (n_boxes, 4)

        if len(bboxes) == 0:
            log_str = f'Failed to ground token with conf_thresh={conf_thresh}'
            logger.warning(log_str)
            return torch.zeros(0, 4).int()

        logger.info(f'Detected {len(bboxes)} bounding boxes')
        bboxes = bboxes.round().int() # (n_detected, 4); Desco returns fractional bbox predictions

        return bboxes

    def desco_mask(self, img: Image, caption: str, tokens_to_ground: list[str], conf_thresh: float = .4) -> torch.BoolTensor:
        '''
            Returns (n_detected, h, w) boolean array. If no detections are returned by DesCo, returns a
            tensor of shape (0, h, w).
        '''
        bboxes = self.desco_ground(img, caption, tokens_to_ground, conf_thresh).numpy() # SAM takes numpy bbox

        if len(bboxes) == 0:
            return torch.zeros(0, img.size[1], img.size[0]).bool() # (0, h, w); Image.Image has size (w, h)

        logger.info('Obtaining segmentation mask with SAM')
        self.sam.set_image(np.array(img))

        grounded_masks = []
        for bbox in  bboxes:
            masks, scores, logits = self.sam.predict(box=bbox, multimask_output=True)
            mask = torch.from_numpy(masks[np.argmax(scores)]).bool()
            grounded_masks.append(mask)

        grounded_masks = torch.stack(grounded_masks) # (n_detected, h, w)

        return grounded_masks

    def localize(self, img: Image, caption='', tokens_to_ground: list[str] = [], conf_thresh: float = .4) -> torch.IntTensor:
        '''
            Returns torch.IntTensor of shape (n_detections, h, w).

            If caption and token_to_ground are provided, uses DesCo to ground the token in the image.
            Otherwise, uses rembg to perform foreground segmentation.
        '''
        img = img.convert('RGB')

        if caption:
            assert all(t in caption for t in tokens_to_ground)
            logger.info(f'Localizing with DesCo to ground "{tokens_to_ground}" with caption "{caption}"')
            bboxes = self.desco_ground(img, caption, tokens_to_ground, conf_thresh)

        else: # rembg
            logger.info('Localizing with rembg')
            bboxes = self.rembg_ground(img)

        return bboxes

# %%
if __name__ == '__main__':
    # Imports
    import PIL
    from vis_utils import show, image_from_masks
    from torchvision.utils import draw_bounding_boxes
    from torchvision.transforms.functional import pil_to_tensor
    from feature_extraction import build_sam_predictor, build_desco
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

    desco_bboxes = localizer.desco_ground(img, caption, token)
    desco_masks = localizer.desco_mask(img, caption, token)

    show([
        draw_bounding_boxes(img_tensor, boxes=desco_bboxes, colors='red'),
        image_from_masks(desco_masks, superimpose_on_image=img_tensor)
    ], title='DesCo')

    rembg_bboxes = localizer.rembg_ground(img)
    rembg_masks = localizer.rembg_mask(img)

    show([
        draw_bounding_boxes(img_tensor, boxes=rembg_bboxes, colors='red'),
        image_from_masks(rembg_masks, superimpose_on_image=img_tensor)
    ], title='Rembg')
# %%
