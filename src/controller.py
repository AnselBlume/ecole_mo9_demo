# %%
if __name__ == '__main__': # TODO Delete me after debugging
    import sys
    sys.path.append('/shared/nas2/blume5/fa23/ecole/src/mo9_demo/src')

from score import Scorer
from model.concept import ConceptDB, Concept
from segment import Segmenter
from localize import Localizer, bbox_from_mask
from PIL.Image import Image
import logging, coloredlogs
import inflect
from predictors import build_sam, build_desco, Sam, GLIPDemo
import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from typing import Union

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

class Controller:
    def __init__(self, sam: Sam, desco: GLIPDemo, concept_db: ConceptDB):
        self.concepts = concept_db

        self.scorer = Scorer()
        self.segmenter = Segmenter(sam)
        self.localizer = Localizer(sam, desco)

        self.p = inflect.engine()

    ##############
    # Prediction #
    ##############
    def predict_image(self, image: Image):
        pass

    @torch.inference_mode()
    def localize_and_segment(
        self,
        image: Image,
        concept_name: str = '',
        concept_parts: list[str] = [],
        remove_background: bool = True,
        return_crops: bool = True,
    ) -> Union[torch.BoolTensor, list[Image]]:
        '''
            Localizes and segments the concept in the image in to parts.

            Arguments:
                image (PIL.Image.Image): Image to localize and segment
                concept_name (str): Name of concept to localize. If not provided, uses rembg to perform foreground segmentation
                concept_parts (list[str]): List of part names to localize. If not provided, uses SAM to perform part segmentation
                remove_background (bool): Whether to remove the background from the localized concept.
                return_crops (bool): If true, returns images of the cropped parts (possibly with background removed).
                    Otherwise, returns the full image parts mask of shape (n_parts, h, w).
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
            logger.info('Localizing concept parts with DesCo')
            # concept = self.concepts.get_concept(concept_name)
            # component_parts = list(concept.component_concepts.keys())
            # TODO consider setting areas not in the bbox to zero instead of cropping the image to maintain scale
            cropped_image = self.segmenter.crop(image, bboxes[0], remove_background=remove_background) # Crop around localized concept
            part_masks = self.localizer.desco_mask(cropped_image, caption=caption, tokens_to_ground=concept_parts) # (n_detections, h, w)

            if len(part_masks) == 0:
                raise RuntimeError('Failed to localize concept parts with DesCo')

            # Convert masks into full image size
            full_part_masks = torch.zeros(len(part_masks), image.size[1], image.size[0], dtype=torch.bool)
            x1, y1, x2, y2 = bboxes[0]

            if remove_background: # Crop had background removed, so just extract mask
                crop_foreground = pil_to_tensor(cropped_image).bool().sum(dim=0).bool()

            else: # Don't remove background
                crop_foreground = torch.ones(cropped_image.size[1], cropped_image.size[0], dtype=torch.bool)

            for i, part_mask in enumerate(part_masks):
                full_part_masks[i, y1:y2, x1:x2] = part_mask & crop_foreground

            part_masks = full_part_masks

        else: # Non part-based segmentation of localized concept
            logger.info('Performing part segmentation with SAM')
            part_masks = self.segmenter.segment(image, bboxes[0], remove_background=remove_background)

        if return_crops:
            return self.segmenter.crops_from_masks(image, part_masks, only_mask=remove_background)

        return part_masks

    def _get_article(self, word: str, space_if_nonempty: bool = True):
        '''
            Returns 'a/an' if the word is singular, else returns ''
        '''
        # Determine the article for the concept
        is_singular_noun = not bool(self.p.singular_noun(word)) # Returns false if singular; singular version if plural
        article = self.p.a(word) if is_singular_noun else ''

        if article: # Split off the 'a' or 'an' from the word
            article = article.split(' ')[0]

        if space_if_nonempty and article:
            article += ' '

        return article

    def _get_parts_caption(self, concept_name: str, component_parts: list[str]):
        '''
            dog, head, whiskers, tail --> a dog with a head, whiskers, and a tail
        '''
        prompt = f'{self._get_article(concept_name)}{concept_name} '
        for i, component_part in enumerate(component_parts):
            if i == 0:
                prompt += 'with '
            elif i == len(component_parts) - 1:
                prompt += ', and ' if len(component_parts) > 2 else ' and '
            else:
                prompt += ', '

            prompt += f'{self._get_article(component_part)}{component_part}'

        return prompt

    ###################
    # Concept Removal #
    ###################
    def clear_concepts(self):
        pass

    def remove_concept(self, concept_name: str):
        pass

    ####################
    # Concept Addition #
    ####################
    def add_concept(self, concept: Concept):
        # Get zero shot attributes (query LLM)

        # Determine if it has any obvious parent or child concepts

        # Get likely learned attributes

        # Add concept

        pass

    def _get_zs_attributes(self, concept_name: str):
        pass

    ########################
    # Concept Modification #
    ########################
    def add_zs_attribute(self, concept_name: str, zs_attr_name: str, weight: float):
        pass

    def remove_zs_attribute(self, concept_name: str, zs_attr_name: str):
        pass

    def add_learned_attribute(self, concept_name: str, learned_attr_name: str, weight: float):
        pass

    def remove_learned_attribute(self, concept_name: str, learned_attr_name: str):
        pass

    ##################
    # Interpretation #
    ##################
    def compare_concepts(self, concept1_name: str, concept2_name: str):
        pass

# %%
if __name__ == '__main__':
    import PIL
    import os
    from torchvision.utils import draw_bounding_boxes
    from vis_utils import show, image_from_masks

    sam = build_sam()
    desco = build_desco()
    controller = Controller(sam, desco, ConceptDB())

    # %% Path
    path = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/dog.png'
    out_path = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/dog_out'
    # path = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/fork_2_9.jpg'
    # out_path = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/fork_out'
    os.makedirs(out_path, exist_ok=True)

    img = PIL.Image.open(path).convert('RGB')

    # %%
    def show_example(concept_name, concept_parts, file_fmt):
        localized = controller.localizer.localize(img, caption=concept_name, tokens_to_ground=[concept_name])
        bbox_img = draw_bounding_boxes(pil_to_tensor(img), localized[0:1], colors='red', width=8)
        fig, ax = show(bbox_img)
        fig.savefig(os.path.join(out_path, file_fmt.format('localized')))

        masks = controller.localize_and_segment(img, concept_name=concept_name, concept_parts=concept_parts, return_crops=False)
        mask_img = image_from_masks(masks, superimpose_on_image=pil_to_tensor(img))
        show(mask_img, title='Segmentation plots')

        crops = controller.localize_and_segment(img, concept_name=concept_name, concept_parts=concept_parts, return_crops=True)
        for crop in crops:
            show(crop, title='Cropped part')
        fig.savefig(os.path.join(out_path, file_fmt.format('segmented')))

    # %% No concept name, no concept parts
    show_example('', [], 'no_name-no_parts-{}.jpg')

    # %% Concept name, no concept parts
    show_example('animal', [], 'name-no_parts-{}.jpg')
    # show_example('fork', [], 'name-no_parts-{}.jpg')

    # %% Concept name, concept parts
    # TODO investigate why this returns so many masks. Does desco output 7 bboxes?
    show_example('animal', ['head', 'ears', 'body'], 'name-parts-{}.jpg')
    # show_example('fork', ['tines', 'handle'], 'name-parts-{}.jpg')
# %%
#
