import torch
import torch.nn.functional as F
from image_processing import LocalizeAndSegmentOutput
from torch.utils.data import DataLoader, Dataset
from model.concept import ConceptKB, Concept
from model.concept_predictor import ConceptPredictorOutput
from .feature_cache import CachedImageFeatures
from kb_ops.dataset import ImageDataset, list_collate, PresegmentedDataset, FeatureDataset
from .feature_pipeline import ConceptKBFeaturePipeline
from typing import Union
from PIL.Image import Image

class ConceptKBForwardBase:
    UNK_LABEL = '[UNK]'

    def __init__(self, concept_kb: ConceptKB, feature_pipeline: ConceptKBFeaturePipeline = None):
        '''
            feature_pipeline must be provided if not using a FeatureDataset.
        '''

        self.concept_kb = concept_kb

        self.label_to_index: dict[str,int] = {concept.name : i for i, concept in enumerate(concept_kb)}
        self.label_to_index[self.UNK_LABEL] = -1 # For unknown labels
        self.index_to_label: dict[int,str] = {v : k for k, v in self.label_to_index.items()}

        self.feature_pipeline = feature_pipeline

    def _get_dataloader(self, dataset: Dataset, is_train: bool, **dl_kwargs):
        kwargs = {
            'batch_size': 1,
            'shuffle': is_train,
            'collate_fn': list_collate,
            'num_workers': 0,
            'pin_memory': True,
            # 'persistent_workers': True
        }

        kwargs.update(dl_kwargs)

        return DataLoader(**kwargs)

    def _determine_data_key(self, dataset: Union[ImageDataset, PresegmentedDataset, FeatureDataset]):
        if isinstance(dataset, FeatureDataset):
            return 'features'

        elif isinstance(dataset, PresegmentedDataset):
            return 'segmentations'

        else:
            assert isinstance(dataset, ImageDataset)
            return 'image'

    def recompute_labels(self):
        '''
            Recomputes label to index (and reverse) mappings based on the current ConceptKB.
        '''
        self.label_to_index = {concept.name : i for i, concept in enumerate(self.concept_kb)}
        self.label_to_index[self.UNK_LABEL] = -1 # For unknown labels
        self.index_to_label = {v : k for k, v in self.label_to_index.items()}

    def forward_pass(
        self,
        image_data: Union[Image, LocalizeAndSegmentOutput, CachedImageFeatures],
        text_label: str = None,
        concepts: list[Concept] = None,
        do_backward: bool = False,
        backward_every_n_concepts: int = None,
        return_segmentations: bool = False,
    ):

        # Check __name__ instead of isinstance to avoid pickle versioning issues
        if image_data.__class__.__name__ == 'CachedImageFeatures':
            features_were_provided = True

        else: # Not using features
            image, segmentations = self.feature_pipeline.get_image_and_segmentations(image_data)
            features_were_provided = False

        # Get all concept predictions
        total_loss = 0
        curr_loss = 0
        outputs = []

        # Cache to avoid recomputation for each image
        cached_visual_features = None
        cached_trained_attr_scores = None

        concepts = concepts if concepts is not None else self.concept_kb
        for i, concept in enumerate(concepts, start=1):
            if features_were_provided:
                device = concept.predictor.img_features_predictor.weight.device
                features = image_data.get_image_features(concept.name).to(device)

            else: # Features not provided; compute from segmentations
                zs_attrs = [attr.query for attr in concept.zs_attributes]

                features = self.feature_pipeline.get_features(
                    image,
                    segmentations,
                    zs_attrs,
                    cached_visual_features=cached_visual_features,
                    cached_trained_attr_scores=cached_trained_attr_scores
                )

                # Cache visual features and trained attribute scores
                if cached_visual_features is None:
                    cached_visual_features = torch.cat([features.image_features, features.region_features], dim=0)

                if cached_trained_attr_scores is None:
                    cached_trained_attr_scores = torch.cat([features.trained_attr_img_scores, features.trained_attr_region_scores], dim=0)

            # Compute concept predictor outputs
            output: ConceptPredictorOutput = concept.predictor(features)
            score = output.cum_score

            # Compute loss and potentially perform backward pass
            if text_label is not None:
                binary_label = torch.tensor(int(concept.name == text_label), dtype=score.dtype, device=score.device)
                concept_loss = F.binary_cross_entropy_with_logits(score, binary_label) / len(self.concept_kb)

                curr_loss += concept_loss
                total_loss += concept_loss.item()

            outputs.append(output.to('cpu'))

            if (
                do_backward and backward_every_n_concepts is not None
                and (i % backward_every_n_concepts == 0 or i == len(concepts))
            ):
                curr_loss.backward()
                curr_loss = 0

        if do_backward and backward_every_n_concepts is None: # Backward if we weren't doing it every K concepts
            curr_loss.backward()

        # Return results
        ret_dict = {
            'loss': total_loss if text_label is not None else None,
            'predictors_outputs': outputs,
            'concept_names': [concept.name for concept in concepts]
        }

        if return_segmentations:
            assert not features_were_provided, 'Cannot return segmentations if features were provided as input'
            ret_dict['segmentations'] = segmentations

        return ret_dict