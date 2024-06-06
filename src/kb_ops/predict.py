import torch
from image_processing import LocalizeAndSegmentOutput
from torch.utils.data import DataLoader
from model.concept import ConceptKB, Concept, ConceptPredictorOutput
from .caching import CachedImageFeatures
from .feature_pipeline import ConceptKBFeaturePipeline
from typing import Union
from tqdm import tqdm
from PIL.Image import Image
from .forward import ConceptKBForwardBase, ForwardOutput
from model.dataclass_base import DictDataClass, DeviceShiftable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__file__)

@dataclass
class PredictOutput(DictDataClass, DeviceShiftable):
    concept_names: list[str] = None
    predictors_scores: torch.Tensor = None
    predicted_index: int = None
    predicted_label: str = None
    is_below_unk_threshold: bool = None
    predicted_concept_outputs: ConceptPredictorOutput = None # This will always be the maximizing concept
    predicted_concept_components_to_scores: dict[str,float] = None # Mapping from predicted concept's components to their scores
    true_index: int = None
    true_concept_outputs: torch.Tensor = None
    segmentations: LocalizeAndSegmentOutput = None

    def to(self, device: torch.device, detach: bool = False):
        DeviceShiftable.to(self, device, detach=detach)
        self.segmentations.to(device, detach=detach)

        return self

class ConceptKBPredictor(ConceptKBForwardBase):
    def __init__(self, concept_kb: ConceptKB, feature_pipeline: ConceptKBFeaturePipeline = None):
        '''
            feature_pipeline must be provided if not using a FeatureDataset.
        '''
        super().__init__(concept_kb, feature_pipeline)

    @torch.inference_mode()
    def hierarchical_predict(
        self,
        predict_dl: DataLoader = None,
        image_data: Union[Image, LocalizeAndSegmentOutput, CachedImageFeatures] = None,
        root_concepts: list[Concept] = None,
        unk_threshold: float = 0.,
        include_component_concepts: bool = False,
        **forward_kwargs
    ) -> Union[list[PredictOutput], list[list[PredictOutput]]]:
        '''
            If image_data is not None, returns a list of prediction dicts delineating the prediction path down through the tree.
            If predict_dl is not None, returns a list of lists of prediction dicts, where each list corresponds to the prediction
                path for each example in the DataLoader.

            The final prediction dictionary in a prediction path may be below the unk_threshold, in which case the prediction
            is "unknown" if the prediction path has length 1, or the prediction is the dictionary at index -2.

            If the final prediction dictionary is above the unk_threshold, the dictionary represents the prediction.
        '''
        if image_data:
            component_concepts = set(self.concept_kb.component_concepts)
            def filter_component_concepts(pool: list[Concept]):
                return [concept for concept in pool if concept not in component_concepts]

            prediction_path = []

            pool = root_concepts if root_concepts else self.concept_kb.root_concepts
            if not include_component_concepts:
                pool = filter_component_concepts(pool)

            # Cache features to avoid resegmentation and recomputation of image features
            forward_kwargs['return_segmentations'] = True
            segmentations = None

            while pool: # While we can go deeper down the hierarchy
                logger.info(f'Considering concept pool in hierarchical_predict: {[concept.name for concept in pool]}')

                prediction = self.predict(
                    image_data=image_data,
                    unk_threshold=unk_threshold,
                    concepts=pool,
                    leaf_nodes_only=False,
                    include_component_concepts=include_component_concepts,
                    **forward_kwargs
                )
                prediction_path.append(prediction)

                if segmentations is None:
                    segmentations: LocalizeAndSegmentOutput = prediction['segmentations']
                    segmentations.input_image = image_data
                    image_data = segmentations

                maximizing_concept = self.concept_kb[prediction['predicted_label']]
                if prediction['is_below_unk_threshold']: # If predicted unknown, stop here
                    break

                pool = maximizing_concept.child_concepts.values()
                if not include_component_concepts:
                    pool = filter_component_concepts(pool)

            return prediction_path

        else: # predict_dl
            assert predict_dl is not None, 'Exactly one of predict_dl or image_data must be provided'
            assert predict_dl.batch_size == 1, 'predict_dl must have batch_size of 1'
            data_key = self._determine_data_key(predict_dl.dataset)

            prediction_paths = []
            for batch in tqdm(predict_dl, desc='Prediction'):
                image, text_label = batch[data_key], batch.get('label', [None])

                prediction_path = self.hierarchical_predict(
                    image_data=image[0],
                    text_label=text_label[0], # Passed to forward_pass
                    unk_threshold=unk_threshold,
                    **forward_kwargs
                )

                prediction_paths.append(prediction_path)

            return prediction_paths

    @torch.inference_mode()
    def predict(
        self,
        predict_dl: DataLoader = None,
        image_data: Union[Image, LocalizeAndSegmentOutput, CachedImageFeatures] = None,
        unk_threshold: float = 0.,
        leaf_nodes_only: bool = True,
        include_component_concepts: bool = False,
        **forward_kwargs
    ) -> Union[list[PredictOutput], PredictOutput]:
        '''
            unk_threshold: Number between [0,1]. If sigmoid(max concept score) is less than this,
                the field 'is_below_unk_threshold' will be set to True in the prediction dict.

            Returns: List of PredictOutputs if predict_dl is provided, else a single PredictOutput.
        '''
        if not ((predict_dl is None) ^ (image_data is None)):
            raise ValueError('Exactly one of predict_dl or image_data must be provided')

        self.concept_kb.eval()
        predictions = []

        if leaf_nodes_only:
            assert not forward_kwargs.get('concepts', None), 'Cannot specify concepts with leaf_nodes_only=True'
            forward_kwargs['concepts'] = self.concept_kb.leaf_concepts

        if not include_component_concepts:
            concepts = forward_kwargs.get('concepts', self.concept_kb)
            component_concepts = set(self.concept_kb.component_concepts)
            concepts_minus_component_concepts = [concept for concept in concepts if concept not in component_concepts]
            forward_kwargs['concepts'] = concepts_minus_component_concepts

        logger.debug(f'Concepts passed from predict to forward: {forward_kwargs["concepts"]}')

        def process_outputs(outputs: ForwardOutput):
            # Compute predictions
            if predict_dl is not None: # Provided with a label
                true_ind = outputs.concept_names.index(text_label[0])

            scores = torch.tensor([output.cum_score for output in outputs.predictors_outputs])

            pred_ind = scores.argmax(dim=0).item() # int
            predicted_concept_outputs: ConceptPredictorOutput = outputs.predictors_outputs[pred_ind].cpu()
            predicted_label = outputs.concept_names[pred_ind]
            predicted_concept_components_to_scores = {
                component_name : outputs.all_concept_scores[component_name]
                for component_name in self.concept_kb[predicted_label].component_concepts
            }

            # Indicate whether max score is below unk_threshold
            is_below_unk_threshold = unk_threshold > 0 and scores[pred_ind].sigmoid() < unk_threshold

            prediction = PredictOutput(
                concept_names=outputs.concept_names,
                predictors_scores=scores,
                predicted_index=pred_ind,
                predicted_label=predicted_label,
                is_below_unk_threshold=is_below_unk_threshold,
                predicted_concept_outputs=predicted_concept_outputs,
                predicted_concept_components_to_scores=predicted_concept_components_to_scores,
                true_index=true_ind if predict_dl is not None else None,
                true_concept_outputs=None if predict_dl is None or true_ind < 0 else outputs.predictors_outputs[true_ind]
            )

            if forward_kwargs.get('return_segmentations', False):
                prediction.segmentations = outputs.segmentations

            predictions.append(prediction.cpu())

        if predict_dl is not None:
            data_key = self._determine_data_key(predict_dl.dataset)

            for batch in tqdm(predict_dl, desc='Prediction'):
                image, text_label = batch[data_key], batch.get('label', [None])
                outputs = self.forward_pass(image[0], text_label[0], **forward_kwargs)
                process_outputs(outputs)

            return predictions

        else: # image_data is not None
            outputs = self.forward_pass(image_data, **forward_kwargs)
            process_outputs(outputs)

            return predictions[0]