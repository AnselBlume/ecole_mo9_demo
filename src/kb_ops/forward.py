import torch
import torch.nn.functional as F
from image_processing import LocalizeAndSegmentOutput
from torch.utils.data import DataLoader, Dataset
from model.concept import ConceptKB, Concept, ConceptPredictorOutput
from model.dataclass_base import DictDataClass
from .caching import CachedImageFeatures
from kb_ops.dataset import ImageDataset, list_collate, PresegmentedDataset, FeatureDataset
from .feature_pipeline import ConceptKBFeaturePipeline
from typing import Union, Optional
from dataclasses import dataclass, field
from PIL.Image import Image

@dataclass
class ForwardOutput(DictDataClass):
    loss: Optional[float] = None

    predictors_outputs: list[ConceptPredictorOutput] = field(
        default=None,
        metadata={'help': 'List of ConceptPredictorOutput objects for each Concept specified in the forward pass\' concepts parameter'}
    )

    concept_names: list[str] = field(
        default=None,
        metadata={'help': 'Names of the concepts for which outputs were computed; corresponds to the order of predictors_outputs'}
    )

    binary_concept_predictions: torch.Tensor = field(
        default=None,
        metadata={'help': 'Binary predictions for each concept in concept_names; shape (n_concepts,)'}
    )

    binary_concept_labels: torch.Tensor = field(
        default=None,
        metadata={'help': 'Binary labels for each concept in concept_names; shape (n_concepts,)'}
    )

    all_concept_scores: dict[str, float] = field(
        default=None,
        metadata={'help': 'Mapping from concept name to its detection score (including component concepts which may not be computed here)'}
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

    @property
    def compute_component_concept_scores_from_concept_predictors(self):
        return not self.feature_pipeline.config.compute_component_concept_scores

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

    def forward_pass(
        self,
        image_data: Union[Image, LocalizeAndSegmentOutput, CachedImageFeatures],
        text_label: Union[str, list[str]] = None,
        concepts: list[Concept] = None,
        do_backward: bool = False,
        backward_every_n_concepts: int = None,
        return_segmentations: bool = False,
        seg_kwargs: dict = {}
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

        cached_features = None # Cache to avoid recomputation for each image

        concepts = list(self.concept_kb) if not concepts else concepts
        concept_scores = {}
        all_concept_scores: dict[str,float] = {} # All scores, including component score inputs, to be output
        concepts_for_forward = self._get_concepts_for_forward_pass(concepts) # NOTE This may output a different (topological) order from concepts list
        concepts_for_loss = set(concepts)

        concept_predictions = []
        concept_labels = []
        concept_names = []

        n_concepts_for_loss_processed = 0 # How many concepts intended for loss computation have been processed
        for concept in concepts_for_forward:
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
                        store_component_concept_scores=self.compute_component_concept_scores_from_concept_predictors
                    )

            # If computing component concept scores from concept predictors, retrieve from stored values
            if self.compute_component_concept_scores_from_concept_predictors:
                if concept.component_concepts:
                    component_concept_scores = torch.stack([
                        concept_scores[component_name] for component_name in concept.component_concepts
                    ], dim=-1).unsqueeze(-2) # (..., 1, n_components)
                else:
                    batch_dims = features.image_features.shape[:-2] # All dims before (1, d_img)
                    component_concept_scores = torch.empty(*batch_dims, 1, 0, device=features.image_features.device)

                features.component_concept_scores = component_concept_scores

            all_concept_scores.update({ # Store component concept scores for output regardless of where they are computed
                component_name : component_score.item() if component_score.numel() == 1 else component_score.tolist()
                for component_name, component_score in zip(concept.component_concepts, features.component_concept_scores)
            })

            # Compute concept predictor outputs
            is_concept_for_loss = concept in concepts_for_loss
            with torch.set_grad_enabled(torch.is_grad_enabled() and is_concept_for_loss):
                output: ConceptPredictorOutput = concept.predictor(features)

            score = output.cum_score # (1,) or (batch_size,)
            concept_scores[concept.name] = score.detach()

            # Compute loss and potentially perform backward pass
            if text_label is not None and is_concept_for_loss:
                # This may be a list of text labels (if batched) or a single text label if not
                text_labels = text_label if isinstance(text_label, list) else [text_label]
                if score.dim() == 0:
                    score = score.unsqueeze(0) # (,) -> (1,)
                assert len(score) == len(text_labels), 'Number of text labels must match number of concept predictor outputs'

                binary_labels = torch.tensor([
                    int(text_label in self.concept_labels[concept.name]) for text_label in text_labels],
                    device=score.device,
                    dtype=score.dtype
                )
                binary_predictions = (score > 0).to(binary_labels.dtype)

                concept_loss = F.binary_cross_entropy_with_logits(score, binary_labels) / len(concepts_for_loss)

                concept_labels.extend(binary_labels)
                concept_predictions.extend(binary_predictions)

                curr_loss += concept_loss
                total_loss += concept_loss.item()

            if is_concept_for_loss:
                outputs.append(output.cpu())
                concept_names.append(concept.name)

            if is_concept_for_loss:
                n_concepts_for_loss_processed += 1

                if (
                    do_backward and backward_every_n_concepts is not None
                    and (n_concepts_for_loss_processed % backward_every_n_concepts == 0 or n_concepts_for_loss_processed == len(concepts_for_loss))
                ):
                    curr_loss.backward()
                    curr_loss = 0

        if do_backward and backward_every_n_concepts is None: # Backward if we weren't doing it every K concepts
            curr_loss.backward()

        # Return results
        all_concept_scores.update({
            concept.name : score.item() if score.numel() == 1 else score.tolist()
            for concept.name, score in concept_scores.items()
        }) # Add scores computed here

        forward_output = ForwardOutput(
            loss=total_loss if text_label is not None else None,
            predictors_outputs=outputs,
            concept_names=concept_names,
            binary_concept_predictions=torch.stack(concept_predictions) if concept_predictions else None,
            binary_concept_labels=torch.stack(concept_labels) if concept_labels else None,
            all_concept_scores=all_concept_scores
        )

        if return_segmentations:
            assert not features_were_provided, 'Cannot return segmentations if features were provided as input'
            forward_output.segmentations = segmentations

        return forward_output

    def batched_forward_pass(
        self,
        image_features: CachedImageFeatures,
        concept: Concept,
        text_labels: list[str] = None,
        do_backward: bool = False
    ):
        return self.forward_pass(
            image_features,
            text_label=text_labels,
            concepts=[concept],
            do_backward=do_backward
        )

    def _get_concepts_for_forward_pass(self, concepts: list[Concept]) -> list[Concept]:
        '''
            The concepts necessary for a forward pass may differ from those necessary for prediction,
            as we need to predict the component concepts before the containing concepts.
        '''
        if self.compute_component_concept_scores_from_concept_predictors:
            concepts = self.concept_kb.get_component_concept_subgraph(concepts)
            concepts = self.concept_kb.in_component_order(concepts)

        return concepts

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
