# %%
import torch
import json
from score import AttributeScorer
from model.concept import ConceptKB, Concept, ConceptExample
from PIL.Image import Image
import logging, coloredlogs
from feature_extraction import FeatureExtractor
from image_processing import LocalizerAndSegmenter, LocalizeAndSegmentOutput
from kb_ops import ConceptKBTrainer, ConceptKBPredictor
from kb_ops.retrieve import CLIPConceptRetriever
from kb_ops.train_test_split import split_from_paths
from kb_ops.dataset import FeatureDataset, extend_with_global_negatives, split_from_concept_kb
from utils import ArticleDeterminer
from typing import Union, Literal
from dataclasses import dataclass
from llm import LLMClient
from score import AttributeScorer
from feature_extraction import CLIPAttributePredictor
from kb_ops.feature_pipeline import ConceptKBFeaturePipeline
from kb_ops.caching import ConceptKBFeatureCacher
from utils import to_device
from kb_ops.build_kb import CONCEPT_TO_ATTRS_PATH
from vis_utils import plot_predicted_classes, plot_image_differences, plot_concept_differences, plot_zs_attr_differences

logger = logging.getLogger(__name__)

@dataclass
class ControllerConfig:
    concept_to_zs_attrs_json_path: str = CONCEPT_TO_ATTRS_PATH
    use_concept_predictors_for_concept_components: bool = True

class Controller:
    def __init__(
        self,
        loc_and_seg: LocalizerAndSegmenter,
        concept_kb: ConceptKB,
        feature_extractor: FeatureExtractor,
        retriever: CLIPConceptRetriever = None,
        cacher: ConceptKBFeatureCacher = None,
        zs_predictor: CLIPAttributePredictor = None,
        config: ControllerConfig = ControllerConfig()
    ):

        self.concepts = concept_kb
        self.feature_pipeline = ConceptKBFeaturePipeline(
            concept_kb,
            loc_and_seg,
            feature_extractor,
            compute_component_concept_scores=not config.use_concept_predictors_for_concept_components
        )
        self.trainer = ConceptKBTrainer(concept_kb, self.feature_pipeline)
        self.predictor = ConceptKBPredictor(concept_kb, self.feature_pipeline)
        self.cacher = cacher if cacher else ConceptKBFeatureCacher(concept_kb, self.feature_pipeline, cache_dir='feature_cache')
        self.config = config

        self.retriever = retriever
        self.llm_client = LLMClient()
        self.attr_scorer = AttributeScorer(zs_predictor)

        # Load external knowledgebase of concepts to zero-shot attributes
        if config.concept_to_zs_attrs_json_path:
            with open(config.concept_to_zs_attrs_json_path) as f:
                self.concept_to_zs_attrs = json.load(f)

        else:
            self.concept_to_zs_attrs = {}

        self.cached_predictions = []
        self.cached_images = []

    ##############
    # Prediction #
    ##############
    def clear_cache(self):
        self.cached_predictions = []
        self.cached_images = []

    def predict_concept(
        self,
        image: Image = None,
        loc_and_seg_output: LocalizeAndSegmentOutput = None,
        unk_threshold: float = .1,
        leaf_nodes_only: bool = True,
        restrict_to_concepts: list[str] = []
    ) -> dict:
        '''
        Predicts the concept of an image and returns the predicted label and a plot of the predicted classes.

        Returns: dict with keys 'predicted_label' and 'plot' of types str and PIL.Image, respectively.
        '''
        # TODO Predict with loc_and_seg_output if provided; for use with modified segmentations/background removals
        self.cached_images.append(image)

        if restrict_to_concepts:
            assert not leaf_nodes_only, 'Specifying concepts to restrict prediction to is only supported when leaf_nodes_only=False.'
            concepts = [self.retrieve_concept(concept_name) for concept_name in restrict_to_concepts]
        else:
            concepts = None

        prediction = self.predictor.predict(
            image_data=image,
            unk_threshold=unk_threshold,
            return_segmentations=True,
            leaf_nodes_only=leaf_nodes_only,
            concepts=concepts
        )

        self.cached_predictions.append(prediction)

        img = plot_predicted_classes(prediction, threshold=unk_threshold, return_img=True)
        predicted_label = prediction['predicted_label'] if not prediction['is_below_unk_threshold'] else 'unknown'

        return {
            'predicted_label': predicted_label,
            'plot': img
        }

    def predict_hierarchical(self, image: Image, unk_threshold: float = .1) -> list[dict]:
        return self.predictor.hierarchical_predict(image_data=image, unk_threshold=unk_threshold)

    def predict_from_subtree(self, image: Image, root_concept_name: str, unk_threshold: float = .1) -> list[dict]:
        root_concept = self.retrieve_concept(root_concept_name)
        return self.predictor.hierarchical_predict(image_data=image, root_concepts=[root_concept], unk_threshold=unk_threshold)

    def predict_root_concept(self, image: Image, unk_threshold: float = .1) -> dict:
        results = self.predictor.hierarchical_predict(image_data=image, root_concepts=[self.concepts.root_concept], unk_threshold=unk_threshold) # list[dict]
        return results[0] # The root concept is the first

    def is_concept_in_image(self, image: Image, concept_name: str, unk_threshold: float = .1) -> bool:
        return self.predict_concept(image, unk_threshold=unk_threshold, leaf_nodes_only=False, restrict_to_concepts=[concept_name])

    def localize_and_segment(
        self,
        image: Image,
        concept_name: str = '',
        concept_parts: list[str] = [],
        remove_background: bool = True,
        return_crops: bool = True,
        use_bbox_for_crops: bool = False
    ):
        return self.feature_pipeline.get_segmentations(
            image=image,
            concept_name=concept_name,
            concept_parts=concept_parts,
            remove_background=remove_background,
            return_crops=return_crops,
            use_bbox_for_crops=use_bbox_for_crops
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

    ################################
    # Concept Addition and Removal #
    ################################
    def add_concept(
        self,
        concept_name: str = None,
        concept: Concept = None,
        parent_concept_names: list[str] = [],
        child_concept_names: list[str] = [],
        component_concept_names: list[str] = [],
        use_singular_name: bool = True
    ):
        if not (bool(concept_name is None) ^ bool(concept is None)):
            raise ValueError('Exactly one of concept_name or concept must be provided.')

        if concept_name is not None: # Normalize the name
            concept_name = concept_name.lower()

            if use_singular_name:
                determiner = ArticleDeterminer()
                concept_name = determiner.to_singular(concept_name)

            concept = Concept(concept_name)

        if concept.name in self.concepts:
            raise ValueError(f'Concept with name "{concept.name}" already exists in the ConceptKB.')

        # Add relations if concept_name was provided instead of Concept object
        if concept_name is not None:
            for parent_name in parent_concept_names:
                parent_concept = self.retrieve_concept(parent_name)
                concept.parent_concepts[parent_name] = parent_concept

            for child_name in child_concept_names:
                child_concept = self.retrieve_concept(child_name)
                concept.child_concepts[child_name] = child_concept

            for component_name in component_concept_names:
                component_concept = self.retrieve_concept(component_name)
                concept.component_concepts[component_name] = component_concept

        # Get zero shot attributes (query LLM)
        self.concepts.init_zs_attrs(
            concept,
            self.llm_client,
            encode_class=self.concepts.cfg.encode_class_in_zs_attr,
            zs_attr_dict=self.concept_to_zs_attrs.get(concept.name, None)
        )

        self.concepts.init_predictor(concept)

        if len(self.concepts) and next(iter(self.concepts)).predictor is not None:
            concept.predictor.to(next(self.concepts.parameters()).device) # Assumes all concepts are on the same device
        else: # Assume there aren't any other initialized concept predictors
            concept.predictor.cuda()

        self.concepts.add_concept(concept)
        self.retriever.add_concept(concept)
        self.trainer.recompute_labels()
        self.predictor.recompute_labels()

        return concept

    def clear_concepts(self):
        for concept in self.concepts:
            self.concepts.remove_concept(concept.name)
            self.retriever.remove_concept(concept.name)

        self.trainer.recompute_labels()
        self.predictor.recompute_labels()

    def remove_concept(self, concept_name: str):
        try:
            concept = self.retrieve_concept(concept_name, max_retrieval_distance=0.)
        except RuntimeError as e:
            logger.info(f'No exact match for concept with name "{concept_name}". Not removing concept to be safe.')
            raise(e)

        self.concepts.remove_concept(concept.name)
        self.retriever.remove_concept(concept.name)
        self.trainer.recompute_labels()
        self.predictor.recompute_labels()

    ########################
    # Concept Modification #
    ########################
    def train(
        self,
        split: tuple[float, float, float] = (.6, .2, .2),
        use_concepts_as_negatives: bool = False
    ):
        '''
            Trains all concepts in the concept knowledge base from each concept's example_imgs.
        '''

        self.cacher.cache_segmentations()
        self.cacher.cache_features()

        # Recache all concepts' zero-shot features in case new concepts were added since last training
        for concept in self.concepts:
            self.cacher.recache_zs_attr_features(concept)

            if not self.config.use_concept_predictors_for_concept_components: # Using fixed scores for concept-image pairs
                self.cacher.recache_component_concept_scores(concept)

        train_ds, val_ds, test_ds = split_from_concept_kb(self.concepts, split=split, use_concepts_as_negatives=use_concepts_as_negatives)

        self.trainer.train(
            train_ds=train_ds,
            val_ds=val_ds,
            n_epochs=15,
            lr=1e-2,
            backward_every_n_concepts=10,
            ckpt_dir=None
        )

    def train_concept(
        self,
        concept_name: str,
        stopping_condition: Literal['n_epochs', 'until_correct'] = 'n_epochs',
        new_examples: list[ConceptExample] = [],
        n_epochs: int = 5,
        sample_all_negatives: bool = False,
        min_prob_margin = .2,
        max_retrieval_distance=.01,
        use_concepts_as_negatives: bool = False
    ):
        '''
            Trains the specified concept with name concept_name for the specified number of epochs.

            Args:
                concept_name: The concept to train. If it does not exist, it will be created.
                stopping_condition: The condition to stop training. Must be 'n_epochs' as 'until_correct' is disabled.
                new_examples: If provided, these examples will be added to the concept's examples list.
                sample_all_negatives: Unused parameter as stopping_condition=='until_correct' is disabled.
        '''
        # Try to retrieve concept
        try:
            concept = self.retrieve_concept(concept_name, max_retrieval_distance=max_retrieval_distance) # Low retrieval distance to force exact match
            logger.info(f'Retrieved concept with name: "{concept.name}"')
        except:
            logger.info(f'No concept found for "{concept_name}". Creating new concept.')
            concept = self.add_concept(concept_name)

        if stopping_condition == 'until_correct':
            if not new_examples:
                logger.info('No examples provided; considering all concept examples as stopping condition via correctness')
                new_examples = concept.examples

        # If new_examples are not already in the Concept, add them to the examples list
        # Identify concept examples by their image_paths
        image_paths = {ex.image_path for ex in concept.examples}
        for example in new_examples:
            if example.image_path not in image_paths:
                concept.examples.append(example)

        # Ensure features are prepared, only generating those which don't already exist or are dirty
        self.cacher.cache_segmentations([concept], only_uncached_or_dirty=True)
        self.cacher.cache_features([concept], only_uncached_or_dirty=True)

        # Hook to recache zs_attr_features after negative examples have been sampled
        # This is faster than calling recache_zs_attr_features on all examples in the concept_kb
        def cache_hook(examples):
            self.cacher.recache_zs_attr_features(concept, examples=examples)

            if not self.config.use_concept_predictors_for_concept_components: # Using fixed scores for concept-image pairs
                self.cacher.recache_component_concept_scores(concept, examples=examples)

        if stopping_condition == 'n_epochs' or len(self.concepts) <= 1:
            if len(self.concepts) == 1:
                logger.info(f'No other concepts in the ConceptKB; training concept in isolation for {n_epochs} epochs.')

            self.trainer.train_concept(
                concept,
                stopping_condition='n_epochs',
                n_epochs=n_epochs,
                post_sampling_hook=cache_hook,
                lr=1e-2,
                use_concepts_as_negatives=use_concepts_as_negatives
            )

        elif stopping_condition == 'until_correct':
            self.trainer.train_concept(
                concept,
                stopping_condition='until_correct',
                min_prob_margin=min_prob_margin,
                new_examples=new_examples,
                sample_all_negatives=sample_all_negatives,
                post_sampling_hook=cache_hook,
                n_epochs_between_predictions=1,
                lr=1e-2,# high learning rate so hopefully doesn't take too long to reach margins
                use_concepts_as_negatives=use_concepts_as_negatives
            )

        else:
            raise ValueError('Unrecognized stopping condition')

    def set_zs_attributes(self, concept_name: str, zs_attrs: list[str]):
        concept = self.retrieve_concept(concept_name)
        concept.zs_attributes = zs_attrs

        self.cacher.recache_zs_attr_features(concept) # Recompute zero-shot attribute scores
        self.train_concept(concept.name, new_examples=concept.examples)

    def retrieve_concept(self, concept_name: str, max_retrieval_distance: float = .5):
        concept_name = concept_name.strip()

        if concept_name in self.concepts:
            return self.concepts[concept_name]

        elif concept_name.lower() in self.concepts:
            return self.concepts[concept_name]

        else:
            retrieved_concept = self.retriever.retrieve(concept_name, 1)[0]
            logger.info(f'Retrieved concept "{retrieved_concept.concept.name}" with distance: {retrieved_concept.distance}')
            if retrieved_concept.distance > max_retrieval_distance:
                raise RuntimeError(f'No concept found for "{concept_name}".')

            return retrieved_concept.concept

    def add_hyponym(self, child_name: str, parent_name: str, child_max_retrieval_distance: float = 0.):
        parent = self.retrieve_concept(parent_name)

        try:
            child = self.retrieve_concept(child_name, max_retrieval_distance=child_max_retrieval_distance)
        except RuntimeError:
            child = self.add_concept(child_name, parent_concept_names=[parent_name])

        parent.child_concepts[child.name] = child
        child.parent_concepts[parent.name] = parent

    def add_component_concept(self, component_concept_name: str, concept_name: str, component_max_retrieval_distance: float = 0.):
        concept = self.retrieve_concept(concept_name)

        try:
            component = self.retrieve_concept(component_concept_name, max_retrieval_distance=component_max_retrieval_distance)
        except RuntimeError:
            component = self.add_concept(component_concept_name)

        concept.component_concepts[component.name] = component
        concept.predictor.set_num_component_concepts(len(concept.component_concepts))

    def add_concept_negatives(self, concept_name: str, negatives: list[ConceptExample]):
        assert all(negative.is_negative for negative in negatives), 'All ConceptExamples must have is_negative=True.'

        concept = self.retrieve_concept(concept_name)
        concept.examples.extend(negatives)

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
    def compare_concepts(self, concept_name1: str, concept_name2: str, weight_by_magnitudes: bool = True):
        concept1 = self.retrieve_concept(concept_name1)
        concept2 = self.retrieve_concept(concept_name2)

        attr_names = self.feature_pipeline.feature_extractor.trained_attr_predictor.attr_names

        return plot_concept_differences(
            (concept1, concept2),
            attr_names,
            weight_by_magnitudes=weight_by_magnitudes,
            take_abs_of_weights=self.concepts.cfg.use_ln,
            return_img=True
        )

    def compare_predictions(
        self,
        indices: tuple[int,int] = None,
        images: tuple[Image,Image] = None,
        weight_by_predictors: bool = True,
        image1_regions: Union[str, list[int]] = None,
        image2_regions: Union[list[str], list[int]] = None
    ) -> Image:
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

        predictions1 = self.cached_predictions[indices[0]]
        predictions2 = self.cached_predictions[indices[1]]

        # Handle case where user wants to visualize region predictions
        if image1_regions or image2_regions:
            def get_region_inds(regions: Union[str, list[int]], predictions: dict):
                part_names = predictions['segmentations'].part_names

                if isinstance(regions, str):
                    inds = [i for i, part_name in enumerate(part_names) if part_name == regions]
                else:
                    inds = regions

                return inds

            def get_region_scores(region_inds: list[int], predictions: dict):
                region_scores = predictions['predicted_concept_outputs'].trained_attr_region_scores[region_inds]
                return region_scores.mean(dim=0) # Average over number of regions

        # Get attr scores and masks for regions or image
        if image1_regions:
            region_inds = get_region_inds(image1_regions, predictions1)
            trained_attr_scores1 = get_region_scores(region_inds, predictions1)
            region_mask1 = predictions1['segmentations'].part_masks[region_inds]
        else:
            trained_attr_scores1 = predictions1['predicted_concept_outputs'].trained_attr_img_scores
            region_mask1 = None

        if image2_regions:
            region_inds = get_region_inds(image2_regions, predictions1)
            trained_attr_scores2 = get_region_scores(region_inds, predictions2)
            region_mask2 = predictions2['segmentations'].part_masks[region_inds]
        else:
            trained_attr_scores2 = predictions2['predicted_concept_outputs'].trained_attr_img_scores
            region_mask2 = None

        # Convert to probabilities if not already
        if not self.concepts.cfg.use_probabilities:
            trained_attr_scores1 = trained_attr_scores1.sigmoid()
            trained_attr_scores2 = trained_attr_scores2.sigmoid()

        # Get attribute names
        attr_names = self.feature_pipeline.feature_extractor.trained_attr_predictor.attr_names

        # Extract winning predictors if weighting by predictors
        if weight_by_predictors:
            predicted_concept1 = predictions1['concept_names'][predictions1['predicted_index']]
            predicted_concept2 = predictions2['concept_names'][predictions2['predicted_index']]

            predictor1 = self.concepts[predicted_concept1].predictor
            predictor2 = self.concepts[predicted_concept2].predictor
            predictors = (predictor1, predictor2)
        else:
            predictors = ()

        return plot_image_differences(
            images,
            (trained_attr_scores1, trained_attr_scores2),
            attr_names,
            weight_imgs_by_predictors=predictors,
            region_masks=(region_mask1, region_mask2),
            return_img=True,
        )

    def compare_zs_attributes(
        self,
        concept_names: tuple[str,str],
        image: Image,
        use_sigmoid: bool = False,
        weight_scores_by_predictors: bool = False
    ):
        if len(concept_names) != 2:
            raise ValueError('concepts must be a tuple of length 2.')

        concept1 = self.retrieve_concept(concept_names[0])
        concept2 = self.retrieve_concept(concept_names[1])

        concept1_zs_attr_names = [a.name for a in concept1.zs_attributes]
        concept2_zs_attr_names = [a.name for a in concept2.zs_attributes]
        all_queries = [a.query for a in concept1.zs_attributes + concept2.zs_attributes]

        # Forward pass through CLIP
        scores = self.feature_pipeline.feature_extractor.zs_attr_predictor.predict([image], all_queries, apply_sigmoid=use_sigmoid) # (1, n_zs_attrs)

        if use_sigmoid:
            scores = scores.sigmoid()

        concept1_scores, concept2_scores = scores[0].split((len(concept1.zs_attributes), len(concept2.zs_attributes)))

        if weight_scores_by_predictors:
            concept1_weights = concept1.predictor.zs_attr_predictor.weight.data.cpu()  # TODO Q) Where do the weights come from?
            concept2_weights = concept2.predictor.zs_attr_predictor.weight.data.cpu()
            predictor_weights = concept1_weights, concept2_weights

        else:
            predictor_weights = ()

        return plot_zs_attr_differences(
            image,
            zs_attr_names=(concept1_zs_attr_names, concept2_zs_attr_names),
            concept_names=(concept1.name, concept2.name),
            concept_scores=(concept1_scores, concept2_scores),
            weight_scores_by_predictors=predictor_weights
        )

    def get_maximizing_region(
        self,
        index: int,
        attr_name: str,
        attr_type: Literal['trained', 'zs'] = 'trained',
        use_abs: bool = False,
        return_all_metadata: bool = False
    ) -> Union[torch.BoolTensor, dict]:
        '''
            Gets the part mask corresponding to the region with the highest score for the
            specified attribute. Assumes the prediction has already been cached.

            Arguments:
                index: Index of the cached prediction to use.
                attr_name: Name of the attribute to visualize.
                attr_type: Type of attribute to visualize. Must be 'trained' or 'zs'.
                use_abs: If True, finds the region which maximizes the (unsigned) magnitude of the score.
                return_all_metadata: If True, returns a dict with keys 'part_masks', 'maximizing_index',
                    and 'attr_scores_by_region'. If False, returns the part mask BoolTensor.

            Returns: If return_all_metadata is False, returns the part mask. Otherwise, returns a dict
                with keys 'part_masks', 'maximizing_index', and 'attr_scores_by_region'.

        '''
        prediction = self.cached_predictions[index]

        if attr_type == 'trained':
            attr_names = self.feature_pipeline.feature_extractor.trained_attr_predictor.attr_names
            attr_scores = prediction['predicted_concept_outputs'].trained_attr_region_scores

        else:
            assert attr_type == 'zs'
            attr_names = [a.name for a in self.concepts[prediction['predicted_label']].zs_attributes]
            attr_scores = prediction['predicted_concept_outputs'].zs_attr_region_scores

        attr_index = attr_names.index(attr_name)
        attr_scores_by_region = attr_scores[:, attr_index] # (n_regions,)

        if use_abs: # Find region which maximizes magnitude of score, regardless of sign
            attr_scores_by_region = attr_scores_by_region.abs()

        maximizing_region_ind = attr_scores_by_region.argmax().item()

        if return_all_metadata:
            return {
                'part_masks': prediction['segmentations'].part_masks,
                'maximizing_index': maximizing_region_ind,
                'attr_scores_by_region': attr_scores_by_region
            }

        else:
            maximizing_region_mask = prediction['segmentations'].part_masks[maximizing_region_ind]
            return maximizing_region_mask

    def explain_prediction(self, index: int = -1):
        # See scripts/vis_contributions.py
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
    ckpt_path = '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_03_22-15:06:03-xob6535d-v3-dino_pool/concept_kb_epoch_50.pt'

    # %%
    kb = ConceptKB.load(ckpt_path)
    loc_and_seg = build_localizer_and_segmenter(build_sam(), build_desco())
    fe = build_feature_extractor()

    retriever = CLIPConceptRetriever(kb.concepts, fe.clip, fe.processor)
    controller = Controller(loc_and_seg, kb, fe, retriever)

    # %% Run the first prediction
    img = PIL.Image.open(img_path).convert('RGB')
    result = controller.predict_concept(img, unk_threshold=.1)

    logger.info(f'Predicted label: {result["predicted_label"]}')

    # %% Run the second prediction
    img_path2 = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/fork_2_9.jpg'
    img2 = PIL.Image.open(img_path2).convert('RGB')
    controller.predict_concept(img2)

    # %% Explain difference between images
    logger.info('Explaining difference between predictions...')
    controller.compare_predictions(indices=(-2,-1), weight_by_predictors=True)

    # %% Explain difference between image regions
    controller.compare_predictions(indices=(-2,-1), weight_by_predictors=True, image1_regions=[0])
    controller.compare_predictions(indices=(-2,-1), weight_by_predictors=True, image2_regions=[0])
    controller.compare_predictions(indices=(-2,-1), weight_by_predictors=True, image1_regions=[0], image2_regions=[0])

    # %% Explain difference between concepts
    controller.compare_concepts('spoon', 'fork')

    # %% Visualize difference between zero-shot attributes
    controller.compare_zs_attributes(('spoon', 'fork'), img)

    # %%
