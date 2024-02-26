# %%
from score import AttributeScorer
from model.concept import ConceptKB, Concept
from PIL.Image import Image
import logging, coloredlogs
from feature_extraction import FeatureExtractor
from image_processing import LocalizerAndSegmenter
from kb_ops.train import ConceptKBTrainer
from llm import LLMClient
from score import AttributeScorer
from feature_extraction import CLIPAttributePredictor
from utils import to_device
from vis_utils import plot_predicted_classes, plot_differences

logger = logging.getLogger(__name__)

class Controller:
    def __init__(
        self,
        loc_and_seg: LocalizerAndSegmenter,
        concept_kb: ConceptKB,
        feature_extractor: FeatureExtractor,
        zs_predictor: CLIPAttributePredictor = None
    ):
        self.concepts = concept_kb
        self.trainer = ConceptKBTrainer(concept_kb, feature_extractor, loc_and_seg)

        self.loc_and_seg = loc_and_seg
        self.llm_client = LLMClient()
        self.attr_scorer = AttributeScorer(zs_predictor)

        self.cached_predictions = []
        self.cached_images = []

    ##############
    # Prediction #
    ##############
    def clear_cache(self):
        self.cached_predictions = []
        self.cached_images = []

    def diff_between_predictions(self, indices: tuple[int,int] = None, images: tuple[Image,Image] = None) -> Image:
        if not ((images is None) ^ (indices is None)):
            raise ValueError('Exactly one of imgs or idxs must be provided.')

        if images is not None:
            if len(images) != 2:
                raise ValueError('imgs must be a tuple of length 2.')

            # Cache predictions
            # NOTE This can be optimized, since running through all concept predictors when only need to
            # 1) Localize and segment
            # 2) Compute image features
            # 3) Compute trained attribute predictions
            self.predict_concept(images[0], unk_threshold=0)
            self.predict_concept(images[1], unk_threshold=0)

            trained_attr_scores1 = self.cached_predictions[-2]['predicted_concept_outputs'].trained_attr_img_scores
            trained_attr_scores1 = self.cached_predictions[-2]['predicted_concept_outputs'].trained_attr_img_scores

        else: # idxs is not None
            if len(indices) != 2:
                raise ValueError('idxs must be a tuple of length 2.')

            images = (self.cached_images[indices[0]], self.cached_images[indices[1]])

            trained_attr_scores1 = self.cached_predictions[indices[0]]['predicted_concept_outputs'].trained_attr_img_scores
            trained_attr_scores2 = self.cached_predictions[indices[1]]['predicted_concept_outputs'].trained_attr_img_scores

        attr_names = self.trainer.feature_extractor.trained_clip_attr_predictor.attr_names

        return plot_differences(*images, trained_attr_scores1, trained_attr_scores2, attr_names, return_img=True)

    def explain_prediction(self, index: int = -1):
        pass

    def predict_concept(self, image: Image, unk_threshold: float = .1) -> dict:
        '''
        Predicts the concept of an image and returns the predicted label and a plot of the predicted classes.

        Returns: dict with keys 'predicted_label' and 'plot' of types str and PIL.Image, respectively.
        '''
        self.cached_images.append(image)

        prediction = self.trainer.predict(
            image_data=image,
            unk_threshold=unk_threshold,
            return_trained_attr_scores=True
        )

        self.cached_predictions.append(prediction)

        img = plot_predicted_classes(prediction, threshold=unk_threshold, return_img=True)
        predicted_label = prediction['predicted_label']
        predicted_label = predicted_label if predicted_label != self.trainer.UNK_LABEL else 'unknown'

        return {
            'predicted_label': predicted_label,
            'plot': img
        }

    def localize_and_segment(
        self,
        image: Image,
        concept_name: str = '',
        concept_parts: list[str] = [],
        remove_background: bool = True,
        return_crops: bool = True
    ):
        return self.loc_and_seg.localize_and_segment(
            image=image,
            concept_name=concept_name,
            concept_parts=concept_parts,
            remove_background=remove_background,
            return_crops=return_crops
        )

    def predict_from_zs_attributes(
        self,
        image: Image,
        concept_name: str = '',
        concept_parts: list[str] = [],
        remove_background: bool = True,
        zs_attrs: list[str] = []
    ) -> dict:
        segmentations = self.loc_and_seg.localize_and_segment(
            image=image,
            concept_name=concept_name,
            concept_parts=concept_parts,
            remove_background=remove_background,
            return_crops=True
        )

        region_crops = segmentations['part_crops']
        if region_crops == []:
            region_crops = [image]

        # Compute zero-shot scores
        part_zs_scores = to_device(self.zs_attr_match_score(region_crops, zs_attrs), 'cpu')
        full_zs_scores = to_device(self.zs_attr_match_score([image], zs_attrs), 'cpu')

        ret_dict = {
            'segmentations': segmentations,
            'scores': {
                'part_zs_scores': part_zs_scores,
                'full_zs_scores': full_zs_scores
            }
        }

        return ret_dict

    ###########
    # Scoring #
    ###########
    def zs_attr_match_score(self, regions: list[Image], zs_attrs: list[str]):
        results = self.attr_scorer.score_regions(regions, zs_attrs)

        raw_scores = results['zs_scores_per_region_raw'] # (n_regions, n_texts)
        weighted_scores = results['zs_scores_per_region_weighted'] # (n_regions, n_texts)

        attr_probs_per_region = weighted_scores.permute(1, 0).softmax(dim=1) # (n_texts, n_regions)
        match_score = weighted_scores.sum().item()

        ret_dict = {
            'raw_scores': raw_scores,
            'weighted_scores': weighted_scores,
            'attr_probs_per_region': attr_probs_per_region,
            'match_score': match_score
        }

        return ret_dict

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
    from feature_extraction import build_feature_extractor, build_sam, build_desco
    from image_processing import build_localizer_and_segmenter

    coloredlogs.install(level=logging.INFO)

    img_path = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/adversarial_spoon.jpg'
    ckpt_path = '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_02_22-00:47:36-s1roip9b-two_scales/concept_kb_epoch_15.pt'

    # %%
    loc_and_seg = build_localizer_and_segmenter(build_sam(), build_desco())
    fe = build_feature_extractor()

    kb = ConceptKB.load(ckpt_path)
    controller = Controller(loc_and_seg, kb, fe)

    # %% Run the prediction
    img = PIL.Image.open(img_path).convert('RGB')
    result = controller.predict_concept(img, unk_threshold=.1)

    logger.info(f'Predicted label: {result["predicted_label"]}')
    result['plot']

    # %% Explain difference
    img_path2 = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/fork_2_9.jpg'
    img2 = PIL.Image.open(img_path2).convert('RGB')
    controller.predict_concept(img2)

    logger.info('Explaining difference between predictions...')
    controller.diff_between_predictions(indices=(-2,-1))

# %%
