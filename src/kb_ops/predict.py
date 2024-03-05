import torch
from image_processing import LocalizeAndSegmentOutput
from torch.utils.data import DataLoader
from model.concept import ConceptKB
from .feature_cache import CachedImageFeatures
from .feature_pipeline import ConceptKBFeaturePipeline
from typing import Union
from tqdm import tqdm
from PIL.Image import Image
from .forward import ConceptKBForwardBase

class ConceptKBPredictor(ConceptKBForwardBase):
    def __init__(self, concept_kb: ConceptKB, feature_pipeline: ConceptKBFeaturePipeline = None):
        '''
            feature_pipeline must be provided if not using a FeatureDataset.
        '''
        super().__init__(concept_kb, feature_pipeline)

    @torch.inference_mode()
    def hierarchical_predict(

    ):
        pass

    @torch.inference_mode()
    def predict(
        self,
        predict_dl: DataLoader = None,
        image_data: Union[Image, LocalizeAndSegmentOutput, CachedImageFeatures] = None,
        unk_threshold: float = 0.,
        **forward_kwargs
    ) -> Union[list[dict], dict]:
        '''
            unk_threshold: Number between [0,1]. If sigmoid(max concept score) is less than this,
                the field 'is_below_unk_threshold' will be set to True in the prediction dict.

            Returns: List of prediction dicts if predict_dl is provided, else a single prediction dict.
        '''
        if not ((predict_dl is None) ^ (image_data is None)):
            raise ValueError('Exactly one of predict_dl or image_data must be provided')

        self.concept_kb.eval()
        predictions = []

        def process_outputs(outputs: dict):
            # Compute predictions
            if predict_dl is not None:
                true_ind = self.label_to_index[text_label[0]] # int

            scores = torch.tensor([output.cum_score for output in outputs['predictors_outputs']])
            pred_ind = scores.argmax(dim=0).item() # int
            predicted_concept_outputs = outputs['predictors_outputs'][pred_ind].cpu()

            # Indicate whether max score is below unk_threshold
            is_below_unk_threshold = unk_threshold > 0 and scores[pred_ind].sigmoid() < unk_threshold

            pred_dict = {
                'concept_names': outputs['concept_names'],
                'predictors_scores': scores.cpu(),
                'predicted_index': pred_ind,
                'predicted_label': self.index_to_label[pred_ind],
                'is_below_unk_threshold': is_below_unk_threshold,
                'predicted_concept_outputs': predicted_concept_outputs, # This will always be maximizing concept
                'true_index': true_ind if predict_dl is not None else None,
                'true_concept_outputs': None if predict_dl is None or true_ind < 0 else outputs['predictors_outputs'][true_ind].cpu()
            }

            if forward_kwargs.pop('return_segmentations', False):
                pred_dict['segmentations'] = outputs['segmentations']

            predictions.append(pred_dict)

        if predict_dl is not None:
            data_key = self._determine_data_key(predict_dl.dataset)

            for batch in tqdm(predict_dl, desc='Prediction'):
                image, text_label = batch[data_key], batch['label']
                outputs = self.forward_pass(image[0], text_label[0], **forward_kwargs)
                process_outputs(outputs)

            return predictions

        else: # image_data is not None
            outputs = self.forward_pass(image_data, **forward_kwargs)
            process_outputs(outputs)

            return predictions[0]