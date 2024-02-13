
##########################
# Segment Anything Model #
##########################
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.modeling import Sam

DEFAULT_SAM_CKPT_PATH = '/shared/nas2/blume5/fa23/ecole/checkpoints/sam/sam_vit_h_4b8939.pth'

def build_sam(model_name: str = 'vit_h', ckpt_path: str = DEFAULT_SAM_CKPT_PATH, device: str = 'cuda') -> Sam:
    return sam_model_registry[model_name](checkpoint=ckpt_path).to(device)

def build_sam_predictor(model: Sam = None):
    if model is None:
        model = build_sam()

    return SamPredictor(model)

def build_sam_amg(model: Sam = None, part_based: bool = False):
    '''
        Part-based settings from part segmentation example of
        https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
    '''
    if model is None:
        model = build_sam()

    if part_based:
        return SamAutomaticMaskGenerator(
            model=model,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )

    else: # Default parameters
        return SamAutomaticMaskGenerator(model)

#########
# DesCo #
#########
# from maskrcnn_benchmark.engine.predictor_FIBER import GLIPDemo
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from maskrcnn_benchmark.config import cfg as BASE_DESCO_CONFIG

DEFAULT_DESCO_CFG_PATH = '/shared/nas2/blume5/fa23/ecole/src/patch_mining/DesCo/configs/pretrain_new/desco_glip.yaml'
DEFAULT_DESCO_CKPT_PATH = '/shared/nas2/blume5/fa23/ecole/checkpoints/desco/desco_glip_tiny.pth'

def build_desco(cfg_path: str = DEFAULT_DESCO_CFG_PATH, ckpt_path: str = DEFAULT_DESCO_CKPT_PATH, device: str = 'cuda'):
    if 'fiber' in (cfg_path + ckpt_path).lower():
        raise NotImplementedError('FIBER GLIPDemo not supported')

    if 'cuda' not in device:
        raise NotImplementedError('DesCo requires GPU')

    cfg = BASE_DESCO_CONFIG.clone()

    cfg.merge_from_file(cfg_path)
    cfg.merge_from_list(['MODEL.WEIGHT', ckpt_path])

    cfg.local_rank = 0 if device == 'cuda' else int(device.split(':')[-1])
    cfg.num_gpus = 1

    desco = GLIPDemo(cfg, min_image_size=800)

    return desco

########
# CLIP #
########
from transformers import CLIPModel, CLIPProcessor

def build_clip(model_name: str = 'openai/clip-vit-large-patch14', device: str = 'cuda') -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    return model, processor

#########################
# Attribute Classifiers #
#########################
from predictors.trained_attrs import TrainedCLIPAttributePredictor
from predictors.clip_features import CLIPFeatureExtractor
from predictors.zero_shot_attrs import CLIPAttributePredictor

def build_trained_attr_predictor(clip_model: CLIPModel, processor: CLIPProcessor, device: str = 'cuda'):
    feature_extractor = CLIPFeatureExtractor(clip_model, processor)
    return TrainedCLIPAttributePredictor(feature_extractor, device=device)

def build_zero_shot_attr_predictor(clip_model: CLIPModel, processor: CLIPProcessor):
    return CLIPAttributePredictor(clip_model, processor)