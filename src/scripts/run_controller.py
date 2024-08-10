import logging

import os
import sys
sys.path = [os.path.realpath(os.path.join(__file__, '../..'))] + sys.path

# Cannot use NFS with mp temp files as otherwise get "device or resource busy" errors
tmp_dir = '/scratch/tmp'
os.makedirs(tmp_dir, exist_ok=True)
os.environ['TMPDIR'] = tmp_dir

from controller import Controller

logger = logging.getLogger(__file__)

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    import PIL
    from feature_extraction import build_feature_extractor, build_sam, build_desco
    from image_processing import build_localizer_and_segmenter
    from model.concept import ConceptKB, ConceptExample
    import PIL.Image
    import coloredlogs
    from kb_ops.feature_pipeline import ConceptKBFeaturePipeline
    import sys
    #from kb_ops.concurrency import LockType

    coloredlogs.install(level=logging.INFO)

    ckpt_path = '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_07_23-02:04:37-all_planes_and_guns_v3-rm_bg_with_component_rem_bg/concept_kb_epoch_313.pt'
    kb = ConceptKB.load(ckpt_path)
    loc_and_seg = build_localizer_and_segmenter(build_sam(), None)
    fe = build_feature_extractor()
    feature_pipeline = ConceptKBFeaturePipeline(loc_and_seg, fe)

    controller = Controller(kb, feature_pipeline)
    controller.heatmap_class_difference('transport plane', 'airplane')














