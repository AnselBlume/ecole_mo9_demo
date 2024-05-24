import torch
import torch.nn.functional as F
from image_processing import LocalizeAndSegmentOutput
from torch.utils.data import DataLoader, Dataset
from model.concept import ConceptKB, Concept, ConceptPredictorOutput
from .feature_cache import CachedImageFeatures
from kb_ops.dataset import ImageDataset, list_collate, PresegmentedDataset, FeatureDataset
from .feature_pipeline import ConceptKBFeaturePipeline
from typing import Union, Optional
from dataclasses import dataclass, field
from PIL.Image import Image

class DictDataClass:
    '''
        Class used for backward compatibility with code that expected dictionary outputs.
    '''
    def __getitem__(self, key):
        return getattr(self, key)

@dataclass
class ForwardOutput(DictDataClass):
    loss: Optional[float] = None

    predictors_outputs: list = field(
        default=None,
        metadata={'help': 'List of ConceptPredictorOutput objects for each Concept specified in the forward pass\' concepts parameter'}
    )

    concept_names: list[str] = field(
        default=None,
        metadata={'help': 'Names of the concepts for which outputs were computed; corresponds to the order of predictors_outputs'}
    )

    segmentations: LocalizeAndSegmentOutput = field(
        default=None,
        metadata={'help': 'Segmentation output if return_segmentations was set to True in the forward pass'}
    )

class ConceptKBForwardBase:
    UNK_LABEL = '[UNK]'

    def __init__(self, concept_kb: ConceptKB, feature_pipeline: ConceptKBFeaturePipeline = None):
        '''
            feature_pipeline must be provided if not using a FeatureDataset.
        '''

        self.concept_kb = concept_kb
        self.recompute_labels()
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

        return DataLoader(dataset, **kwargs)

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
            Additionally computes the rooted subtree for each concept to serve as positive labels.
        '''
        self.label_to_ind = {concept.name : i for i, concept in enumerate(self.concept_kb)}
        self.label_to_ind[self.UNK_LABEL] = -1 # For unknown labels
        self.ind_to_label = {v : k for k, v in self.label_to_ind.items()}

        # Compute rooted subtree (descendants + curr node) for each concept to serve as positive labels
        self.concept_labels = {
            concept.name: {c.name for c in self.concept_kb.rooted_subtree(concept)}
            for concept in self.concept_kb
        }

        # Compute mapping between global and leaf indices
        self.leaf_name_to_leaf_ind = {c.name : i for i, c in enumerate(self.concept_kb.leaf_concepts)}
        self.leaf_ind_to_leaf_name = {v : k for k, v in self.leaf_name_to_leaf_ind.items()}

        self.global_ind_to_leaf_ind = {
            global_ind : self.leaf_name_to_leaf_ind[concept.name]
            for global_ind, concept in enumerate(self.concept_kb.concepts)
            if concept.name in self.leaf_name_to_leaf_ind
        }
        self.leaf_ind_to_global_ind = {v : k for k, v in self.global_ind_to_leaf_ind.items()}

    def forward_pass(
        self,
        image_data: Union[Image, LocalizeAndSegmentOutput, CachedImageFeatures],
        text_label: str = None,
        concepts: list[Concept] = None,
        do_backward: bool = False,
        backward_every_n_concepts: int = None,
        return_segmentations: bool = False,
        seg_kwargs: dict = {},
        set_score_to_zero: bool = False
    ) -> ForwardOutput:

        # Check __name__ instead of isinstance to avoid pickle versioning issues
        if image_data.__class__.__name__ == 'CachedImageFeatures':
            features_were_provided = True

        else: # Not using features
            image, segmentations = self.feature_pipeline.get_image_and_segmentations(image_data, **seg_kwargs)
            features_were_provided = False

        # Get all concept predictions
        total_loss = 0
        curr_loss = 0
        outputs = []

        # Cache to avoid recomputation for each image
        cached_features = None

        # TODO store the component concept scores
        is_computing_component_concept_scores_from_concept_predictors = not self.feature_pipeline.compute_component_concept_scores

        concepts = concepts if concepts else self.concept_kb
        for i, concept in enumerate(concepts, start=1):
            if features_were_provided:
                device = concept.predictor.img_features_predictor.weight.device
                features = image_data.get_concept_predictor_features(concept.name).to(device)

            else: # Features not provided; compute from segmentations
                features = self.feature_pipeline.get_concept_predictor_features(
                    image,
                    segmentations,
                    concept,
                    cached_features=cached_features
                )

                # Cache visual features and trained attribute scores
                if cached_features is None:
                    cached_features = CachedImageFeatures.from_image_features(features)
                    cached_features.update_concept_predictor_features( # Store zero-shot attributes and potentially component concept scores
                        concept,
                        features,
                        store_component_concept_scores=not is_computing_component_concept_scores_from_concept_predictors
                    )

            if is_computing_component_concept_scores_from_concept_predictors:
                # TODO set from stored value
                pass

            # Compute concept predictor outputs
            output: ConceptPredictorOutput = concept.predictor(features)
            score = output.cum_score

            # Compute loss and potentially perform backward pass
            if text_label is not None:
                binary_label = torch.tensor(int(text_label in self.concept_labels[concept.name]), dtype=score.dtype, device=score.device)
                concept_loss = F.binary_cross_entropy_with_logits(score, binary_label) / len(self.concept_kb)

                # To prevent loss saturation due to sigmoid
                if set_score_to_zero:
                    score = output.cum_score - output.cum_score.detach()
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
        forward_output = ForwardOutput(
            loss=total_loss if text_label is not None else None,
            predictors_outputs=outputs,
            concept_names=[concept.name for concept in concepts]
        )

        if return_segmentations:
            assert not features_were_provided, 'Cannot return segmentations if features were provided as input'
            forward_output.segmentations = segmentations

        return forward_output