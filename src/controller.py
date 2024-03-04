# %%
from score import AttributeScorer
from model.concept import ConceptKB, Concept, ConceptExample
from PIL.Image import Image
import logging, coloredlogs
from feature_extraction import FeatureExtractor
from image_processing import LocalizerAndSegmenter
from kb_ops.train import ConceptKBTrainer
from kb_ops.retrieve import CLIPConceptRetriever
from llm import LLMClient
from score import AttributeScorer
from feature_extraction import CLIPAttributePredictor
from kb_ops.feature_pipeline import ConceptKBFeaturePipeline
from kb_ops.feature_cache import ConceptKBFeatureCacher
from utils import to_device
from vis_utils import plot_predicted_classes, plot_differences

logger = logging.getLogger(__name__)

class Controller:
    def __init__(
        self,
        loc_and_seg: LocalizerAndSegmenter,
        concept_kb: ConceptKB,
        feature_extractor: FeatureExtractor,
        retriever: CLIPConceptRetriever = None,
        cache_dir: str = 'feature_cache',
        zs_predictor: CLIPAttributePredictor = None
    ):
        self.concepts = concept_kb
        self.feature_pipeline = ConceptKBFeaturePipeline(concept_kb, loc_and_seg, feature_extractor)
        self.trainer = ConceptKBTrainer(concept_kb, self.feature_pipeline)
        self.cacher = ConceptKBFeatureCacher(concept_kb, self.feature_pipeline, cache_dir=cache_dir)

        self.retriever = retriever
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
        return self.feature_pipeline.get_segmentations(
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
        segmentations = self.feature_pipeline.get_segmentations(
            image=image,
            concept_name=concept_name,
            concept_parts=concept_parts,
            remove_background=remove_background,
            return_crops=True
        )

        region_crops = segmentations.part_crops
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
        for concept in self.concepts:
            self.concepts.remove_concept(concept.name)

    def remove_concept(self, concept_name: str):
        try:
            concept = self.retrieve_concept(concept_name, max_retrieval_distance=0.)
        except RuntimeError as e:
            logger.info(f'No exact match for concept with name "{concept_name}". Not removing concept to be safe.')
            raise(e)

        self.concepts.remove_concept(concept.name)

    ####################
    # Concept Addition #
    ####################
    def add_concept(self, concept_name: str = None, concept: Concept = None):
        if concept_name is None and concept is None:
            raise ValueError('Either concept_name or concept must be provided.')

        if concept is None:
            concept = Concept(concept_name)

        # Get zero shot attributes (query LLM)
        self.concepts.init_zs_attrs(
            concept,
            self.llm_client,
            encode_class=self.concepts.cfg.encode_class_in_zs_attr
        )

        self.concepts.init_predictor(concept)

        # TODO Determine if it has any obvious parent or child concepts

        self.concepts.add_concept(concept)
        concept.predictor.cuda() # Assumes all other predictors are also on cuda
        self.trainer.recompute_labels()

    def _get_zs_attributes(self, concept_name: str):
        pass

    ########################
    # Concept Modification #
    ########################
    def train(self):
        '''
            Trains all concepts in the concept knowledge base from each concept's example_imgs.
        '''
        pass

    def train_concept(self, concept_name: str, until_correct_examples: list[ConceptExample] = []):
        '''
            Retrains the concept until it correctly predicts the given example images.
        '''
        # Try to retrieve concept
        try:
            concept = self.retrieve_concept(concept_name, max_retrieval_distance=.3)
            logger.info(f'Retrieved concept with name: "{concept.name}"')
        except:
            logger.info(f'No concept found for "{concept_name}". Creating new concept.')
            concept = Concept(concept_name.lower())
            self.add_concept(concept)

        # If until_correct_examples are not already in the Concept, add them to the examples list
        if not until_correct_examples:
            logger.info('No examples provided; considering all concept examples as stopping condition via correctness')

        else: # Identify concept examples by their image_paths
            image_paths = {ex.image_path for ex in concept.examples}
            for example in until_correct_examples:
                if example.image_path not in image_paths:
                    concept.examples.append(example)

        # Ensure features are prepared
        self.cacher.cache_features([concept]) # Ensure features are generated

        # Hook to recache zs_attr_features after negative examples have been sampled
        # This is faster than calling recache_zs_attr_features on all examples in the concept_kb
        cache_hook = lambda exs: self.cacher.recache_zs_attr_features(concept, examples=exs)

        # Train concept
        self.trainer.train_concept(
            concept,
            stopping_condition='until_correct',
            until_correct_examples=until_correct_examples,
            sample_all_negatives=False,
            post_sampling_hook=cache_hook,
            n_epochs_between_predictions=7,
            lr=1e-2 # high learning rate so hopefully doesn't take too long to reach margins
        )

    def set_zs_attributes(self, concept_name: str, zs_attrs: list[str]):
        concept = self.retrieve_concept(concept_name)
        concept.zs_attributes = zs_attrs

        self.cacher.recache_zs_attr_features(concept) # Recompute zero-shot attribute scores
        self.train_concept(concept.name, until_correct_examples=concept.examples)

    def retrieve_concept(self, concept_name: str, max_retrieval_distance: float = .5):
        concept_name = concept_name.strip()

        if concept_name in self.concepts:
            return self.concepts[concept_name]

        elif concept_name.lower() in self.concepts:
            return self.concepts[concept_name]

        else:
            retrieved_concept = self.retriever.retrieve(concept_name, 1)[0]
            logger.debug(f'Retrieved concept "{retrieved_concept.concept.name}" with distance: {retrieved_concept.distance}')
            if retrieved_concept.distance > max_retrieval_distance:
                raise RuntimeError(f'No concept found for "{concept_name}".')

            return retrieved_concept.concept

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
    def compare_concepts(self, concept_name1: str, concept_name2: str):
        concept1 = self.retrieve_concept(concept_name1)
        concept2 = self.retrieve_concept(concept_name2)

        predictor1 = concept1.predictor
        predictor2 = concept2.predictor

        weights1 = predictor1.zs_attr_predictor.weight.data
        weights2 = predictor2.zs_attr_predictor.weight.data

    def compare_predictions(self, indices: tuple[int,int] = None, images: tuple[Image,Image] = None) -> Image:
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

            indices = (-2, -1)

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

# %%
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    import PIL
    from feature_extraction import build_feature_extractor, build_sam, build_desco
    from image_processing import build_localizer_and_segmenter

    coloredlogs.install(level=logging.INFO)

    # %%
    img_path = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/adversarial_spoon.jpg'
    ckpt_path = '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_02_29-01:56:41-b1fr1pbu-new_conceptkb/concept_kb_epoch_15.pt'

    # %%
    loc_and_seg = build_localizer_and_segmenter(build_sam(), build_desco())
    fe = build_feature_extractor()

    kb = ConceptKB.load(ckpt_path)
    retriever = CLIPConceptRetriever(kb.concepts, fe.clip, fe.processor)
    controller = Controller(loc_and_seg, kb, fe, retriever)

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
    controller.compare_predictions(indices=(-2,-1))

# %%
