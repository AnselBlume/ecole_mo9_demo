from base import BaseController
from image_processing import LocalizeAndSegmentOutput
from kb_ops import ConceptKBTrainer, ConceptKBPredictor
from kb_ops.predict import PredictOutput
from visualization.vis_utils import plot_predicted_classes
from PIL.Image import Image

class ControllerPredictionMixin(BaseController):
    ##############
    # Prediction #
    ##############
    def predict_concept(
        self,
        image: Image = None,
        loc_and_seg_output: LocalizeAndSegmentOutput = None,
        unk_threshold: float = .1,
        leaf_nodes_only: bool = True,
        include_component_concepts: bool = False,
        restrict_to_concepts: list[str] = []
    ) -> dict:
        '''
        Predicts the concept of an image and returns the predicted label and a plot of the predicted classes.

        Returns: dict with keys 'predicted_label' and 'plot' of types str and PIL.Image, respectively.
        '''
        # TODO Predict with loc_and_seg_output if provided; for use with modified segmentations/background removals
        if self.config.cache_predictions:
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
            include_component_concepts=include_component_concepts,
            concepts=concepts
        )

        if self.config.cache_predictions:
            self.cached_predictions.append(prediction)

        img = plot_predicted_classes(prediction, threshold=unk_threshold, return_img=True)
        predicted_label = prediction['predicted_label'] if not prediction['is_below_unk_threshold'] else 'unknown'

        return {
            'predicted_label': predicted_label,
            'predict_output': prediction,
            'plot': img
        }

    def predict_hierarchical(self, image: Image, unk_threshold: float = .1, include_component_concepts: bool = False) -> list[dict]:
        prediction_path: list[PredictOutput] = self.predictor.hierarchical_predict(
            image_data=image,
            unk_threshold=unk_threshold,
            include_component_concepts=include_component_concepts
        )

        if prediction_path[-1]['is_below_unk_threshold']:
            predicted_label = 'unknown' if len(prediction_path) == 1 else prediction_path[-2]['predicted_label']
            concept_path = [pred['predicted_label'] for pred in prediction_path[:-1]]
        else:
            predicted_label = prediction_path[-1]['predicted_label']
            concept_path = [pred['predicted_label'] for pred in prediction_path]

        return {
            'prediction_path': prediction_path, # list[PredictOutput]
            'concept_path': concept_path, # list[str]
            'predicted_label': predicted_label # str
        }

    def predict_from_subtree(self, image: Image, root_concept_name: str, unk_threshold: float = .1) -> list[dict]:
        root_concept = self.retrieve_concept(root_concept_name)
        return self.predictor.hierarchical_predict(image_data=image, root_concepts=[root_concept], unk_threshold=unk_threshold)

    def predict_root_concept(self, image: Image, unk_threshold: float = .1, include_component_concepts: bool = False) -> dict:
        results = self.predictor.hierarchical_predict(
            image_data=image,
            root_concepts=[self.concept_kb.root_concepts],
            unk_threshold=unk_threshold,
            include_component_concepts=include_component_concepts
        ) # list[dict]

        return results[0] # The root concept is the first

    def is_concept_in_image(self, image: Image, concept_name: str, unk_threshold: float = .1) -> bool:
        do_localize = self.feature_pipeline.loc_and_seg.config.do_localize
        self.feature_pipeline.loc_and_seg.config.do_localize = False

        prediction = self.predict_concept(
            image,
            unk_threshold=unk_threshold,
            leaf_nodes_only=False,
            restrict_to_concepts=[concept_name],
            include_component_concepts=True
        )

        self.feature_pipeline.loc_and_seg.config.do_localize = do_localize

        return prediction