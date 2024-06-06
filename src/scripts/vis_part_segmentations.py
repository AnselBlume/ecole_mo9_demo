import PIL
import os
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from visualization.vis_utils import image_from_masks
from feature_extraction import build_sam, build_desco
from llm import LLMClient, retrieve_parts
from image_processing import build_localizer_and_segmenter
import logging, coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

if __name__ == '__main__':
    sam = build_sam()
    desco = build_desco()
    llm_client = LLMClient()
    loc_and_seg = build_localizer_and_segmenter(sam, desco)

    # %% Path
    in_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/graduate_descent'
    out_dir = '/shared/nas2/blume5/fa23/ecole/results/2_11_24-graduate_descent'

    os.makedirs(out_dir, exist_ok=True)

    # %%
    def run_segmentation(concept_name, concept_parts, file_fmt, save_crops=False):
        try:
            result = loc_and_seg.localize_and_segment(img, concept_name=concept_name, concept_parts=concept_parts)

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