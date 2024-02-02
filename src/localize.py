'''
    Functions to localize the object region in an image.
'''
import numpy as np
from rembg import remove, new_session
from PIL.Image import Image
from maskrcnn_benchmark.config import cfg as BASE_CONFIG
# from maskrcnn_benchmark.engine.predictor_FIBER import GLIPDemo
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from segment_anything import sam_model_registry, SamPredictor
from dataclasses import dataclass
import logging

logger = logging.getLogger(__file__)

if __name__ == '__main__':
    import coloredlogs
    coloredlogs.install(level=logging.INFO, logger=logger)

@dataclass
class LocalizerConfig:
    rembg_model_name: str = 'sam_prompt'

    desco_cfg_path: str = '/shared/nas2/blume5/fa23/ecole/src/patch_mining/DesCo/configs/pretrain_new/desco_glip.yaml'
    desco_ckpt_path: str = '/shared/nas2/blume5/fa23/ecole/checkpoints/desco/desco_glip_tiny.pth'
    desco_device_rank: int = 0

    sam_model_type: str = 'vit_h'
    sam_ckpt_path: str = '/shared/nas2/blume5/fa23/ecole/checkpoints/sam/sam_vit_h_4b8939.pth'
    sam_device_rank: int = 0

class Localizer:
    def __init__(self, config: LocalizerConfig):
        self.config = config

        self.rembg_session = new_session(model_name=config.rembg_model_name)
        self.desco = self._init_desco()
        self.sam = self._init_sam()

    def _init_desco(self):
        '''
            Configuration based on DesCo repo's run_demo.py.
        '''
        if 'fiber' in (self.config.desco_cfg_path + self.config.desco_ckpt_path).lower():
            raise NotImplementedError('FIBER GLIPDemo not supported')

        cfg = BASE_CONFIG.clone()

        cfg.merge_from_file(self.config.desco_cfg_path)
        cfg.merge_from_list(['MODEL.WEIGHT', self.config.desco_ckpt_path])
        cfg.local_rank = self.config.desco_device_rank
        cfg.num_gpus = 1

        desco = GLIPDemo(cfg, min_image_size=800)

        return desco

    def _init_sam(self):
        sam_model = sam_model_registry[self.config.sam_model_type](checkpoint=self.config.sam_ckpt_path)
        sam_model.to(f'cuda:{self.config.sam_device_rank}')
        sam = SamPredictor(sam_model)

        return sam

    def rembg_mask(self, img: Image):
        return remove(img, session=self.session, post_process_mask=True, only_mask=True)

    def desco_mask(self, img: Image, caption: str, token_to_ground: str, conf_thresh: float):
        '''
            Based on the contents of predictor_glip.py.GLIPDemo.run_on_web_image.
            This function signature must be changed if we wish to use the GLIPDemo from predictor_FIBER.py

            XXX The load method of run_demo.py maps the image to BGR, but this seems to conflict with
            GLIPDemo.build_transform's to_bgr_transform. We therefore ignore run_demo.py and don't load in BGR.
        '''
        # Get bounding box with DesCo
        logger.info('Obtaining bounding box with DesCo')

        caption = caption.lower()
        token_to_ground = token_to_ground.lower()

        predictions = self.desco.compute_prediction(np.array(img), caption, specified_tokens=[token_to_ground])
        top_predictions = self.desco._post_process(predictions, conf_thresh)

        boxes = top_predictions.bbox.numpy() # (n_boxwes, 4)

        if len(boxes) == 0:
            log_str = f'Failed to ground token with conf_thresh={conf_thresh}'
            logger.warning(log_str)
            raise RuntimeError(log_str)

        logger.info(f'Detected {len(boxes)} bounding boxes; taking the first one')

        # SAM
        logger.info('Obtaining segmentation mask with SAM')
        masks, scores, logits = self.sam.predict(
            box=boxes[0],
            multimask_output=True
        )
        mask = masks[np.argmax(scores)]

        return mask

    def localize(self, img: Image, caption='', token_to_ground: str = '', conf_thresh: float = .4):
        '''
            Returns boolean array of shape (h, w)
        '''
        img = img.convert('RGB')

        if caption:
            assert token_to_ground and token_to_ground in caption
            logger.info(f'Localizing with DesCo to ground "{token_to_ground}" with caption "{caption}"')
            mask = self.desco_mask(img, caption, token_to_ground, conf_thresh)

        else: # rembg
            logger.info('Localizing with rembg')
            mask = self.rembg_mask(img)

        return mask

if __name__ == '__main__':
    # TODO Test locally
    pass