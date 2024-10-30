import os
import torch
from model.concept import ConceptKB, Concept, ConceptExample
from wandb.sdk.wandb_run import Run
from kb_ops.dataset import FeatureDataset, extend_with_global_negatives
from kb_ops.feature_pipeline import ConceptKBFeaturePipeline
from typing import  Optional
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import numpy as np
from tqdm import tqdm
from kb_ops.forward import ConceptKBForwardBase
from kb_ops.example_sampler import ConceptKBExampleSampler
from .outputs import ValidationOutput
import logging

logger = logging.getLogger(__file__)

class ConceptKBTrainerBase(ConceptKBForwardBase):
    def __init__(self, concept_kb: ConceptKB, feature_pipeline: ConceptKBFeaturePipeline = None, wandb_run: Run = None):
        '''
            feature_pipeline must be provided if not using a FeatureDataset.
        '''
        super().__init__(concept_kb, feature_pipeline)

        self.sampler = ConceptKBExampleSampler(concept_kb)
        self.run = wandb_run
        self.rng = np.random.default_rng(42)

    def _get_ckpt_path(self, ckpt_dir: str, ckpt_fmt: str, epoch: int):
        return os.path.join(ckpt_dir, ckpt_fmt.format(epoch=epoch))

    def _get_concepts_to_train(self, batch: dict, global_concepts: dict[str, Concept]) -> Optional[list[Concept]]:
        '''
            Gets the concepts to train a given (single-example) batch of examples on.
            Intersects the concepts to train with the global concepts if concepts_to_train is not None.
            Else, uses all global concepts.

            Returns None if no concepts to train after intersection with global concepts.
        '''
        concepts_to_train: Optional[list[str]] = batch['concepts_to_train'][0] # Assuming batch size of 1

        if concepts_to_train: # Not None and not []
            concepts_to_train = [global_concepts[c_name] for c_name in concepts_to_train if c_name in global_concepts]

            if not concepts_to_train:
                return None
        else:
            concepts_to_train = list(global_concepts.values())

        return concepts_to_train

    def construct_dataset_for_concept_training(
        self,
        concept: Concept,
        *, # Force the use of kwargs after this point due to API changes
        use_descendants_as_positives: bool = True,
        n_sampled_positives_per_descendant: int = 3,
        use_concepts_as_negatives: bool = True,
        sample_all_negatives: bool = False,
        sample_only_siblings_for_negatives: bool = True,
        sample_only_leaf_nodes_for_negatives: bool = False,
        use_containing_concepts_for_positives: bool = False,
        n_sampled_positives_per_containing_concept: int = 3,
        n_global_negatives: int = 250
    ) -> tuple[list[ConceptExample], FeatureDataset]:
        '''
            Constructs a dataset for training a single concept by sampling examples.

            This differs from full-blown, offline training scripts where the set of concepts are restricted to one
            in that this method sample examples (and negatives) to create a smaller representative dataset,
            allowing for faster training than offline training using all examples.

            Arguments:
                concept: Concept to train.

                use_descendants_as_positives: Whether to use descendants of the concept as positive examples.
                n_sampled_positives_per_descendant: Number of positive examples to sample per descendant.

                use_concepts_as_negatives: Whether to use negative examples from other concepts in the ConceptKB.

                sample_all_negatives: Whether to sample all negative examples from the negative concepts.
                sample_only_siblings_for_negatives: Whether to sample only siblings of the concept for negative examples.
                sample_only_leaf_nodes_for_negatives: Whether to sample only leaf nodes for negative examples.

                use_containing_concepts_for_positives: Whether to use containing concepts for positive examples.
                n_sampled_positives_per_containing_concept: Number of positive examples to sample per containing concept.

                n_global_negatives: Number of global negative examples to sample for training.

            Returns:
                Tuple of list of ConceptExamples used to train the concept and the constructed FeatureDataset.
        '''
        # Create examples and labels

        # Construct positive examples
        pos_examples = concept.examples
        concept_names = [concept.name] * len(pos_examples)

        if use_descendants_as_positives:
            descendants = self.concept_kb.rooted_subtree(concept)
            descendants = [c for c in descendants if c.name != concept.name] # Exclude self

            descendant_pos_examples, descendant_concept_names = self.sampler.sample_examples(
                descendants,
                n_examples_per_concept=n_sampled_positives_per_descendant
            )

            pos_examples.extend(descendant_pos_examples)
            concept_names.extend(descendant_concept_names)

        if use_containing_concepts_for_positives and concept.containing_concepts:
            # This is a component concept actively contained in another concept (not just a descendant of a component)
            containing_concept_positives, containing_concept_names = self.sampler.sample_examples(
                concept.containing_concepts.values(),
                n_examples_per_concept=n_sampled_positives_per_containing_concept
            )

            pos_examples.extend(containing_concept_positives)
            concept_names.extend(containing_concept_names)

        pos_labels = [ # Handle concept-specific negatives
            concept_name if not ex.is_negative else FeatureDataset.NEGATIVE_LABEL
            for ex, concept_name in zip(pos_examples, concept_names)
        ]

        if not pos_examples:
            logger.warning(f'No positive examples found for concept {concept.name}; skipping')
            return

        # Potentially sample negative concepts
        if use_concepts_as_negatives:
            if sample_only_siblings_for_negatives:

                if concept.parent_concepts: # Not a root node
                    siblings = {} # Not including self
                    for parent in concept.parent_concepts.values():
                        for child in parent.child_concepts.values():
                            if child.name not in siblings and child.name != concept.name:
                                siblings[child.name] = child

                else: # Root node; if this isn't a component concept, sample from other non-component concepts
                    component_concepts = set(self.concept_kb.component_concepts)

                    if concept not in component_concepts:
                        non_component_root_siblings = {
                            c.name : c
                            for c in self.concept_kb.root_concepts
                            if c.name != concept.name and c not in component_concepts
                        }

                        siblings = non_component_root_siblings

                    else:
                        siblings = {}

                neg_concepts = list(siblings.values())

            elif sample_only_leaf_nodes_for_negatives:
                # Leaf nodes which aren't components
                component_concepts = set(self.concept_kb.component_concepts)
                neg_concepts = [c for c in self.concept_kb.leaf_concepts if c not in component_concepts and c != concept]
            else:
                neg_concepts = [c for c in self.concept_kb if c != concept]

            # Sample the negative examples from the negative concepts
            if sample_all_negatives:
                neg_examples, neg_concept_names = self.sampler.get_all_examples(neg_concepts)
            else:
                neg_examples, neg_concept_names = self.sampler.sample_negative_examples(len(pos_examples), neg_concepts)

        else:
            neg_examples = []
            neg_concept_names = []

        # Merge positive and negative examples
        all_samples = pos_examples + neg_examples
        all_labels = pos_labels + neg_concept_names

        # Assume features are already cached and get paths
        all_feature_paths = [sample.image_features_path for sample in all_samples]
        if any(feature_path is None for feature_path in all_feature_paths):
            raise RuntimeError('All examples must have image_features_path set to train individual Concept')

        # Construct train ds
        # No need to restrict concepts to train via dataset since restrict via forward(concepts=[concept])
        train_ds = FeatureDataset(all_feature_paths, all_labels, train_all_concepts_if_unspecified=True)

        # Sample and add global negatives
        global_negatives = self.concept_kb.global_negatives
        global_negatives = self.rng.choice(global_negatives, min(n_global_negatives, len(global_negatives)), replace=False).tolist()

        all_samples += global_negatives

        extend_with_global_negatives(train_ds, global_negatives)

        return all_samples, train_ds

    @torch.inference_mode()
    def validate(self, val_dl: DataLoader, leaf_nodes_only_for_accuracy=True, **forward_kwargs):
        self.concept_kb.eval()

        total_loss = 0
        data_key = self._determine_data_key(val_dl.dataset)

        concepts_for_forward = {c.name : c for c in self.concept_kb}

        # Compute accuracy separately for component and non-component concepts
        component_concepts: dict[str,Concept] = {c.name : c for c in self.concept_kb.component_concepts}
        non_component_concepts: dict[str, Concept] = {c.name : c for c in self.concept_kb if c.name not in component_concepts}

        if leaf_nodes_only_for_accuracy:
            leaf_concepts = {c.name : c for c in self.concept_kb.leaf_concepts}
            component_concepts = {c.name : c for c in component_concepts.values() if c.name in leaf_concepts}
            non_component_concepts = {c.name : c for c in non_component_concepts.values() if c.name in leaf_concepts}

        component_accuracy = Accuracy(task='binary').cuda()
        non_component_accuracy = Accuracy(task='binary').cuda()

        for batch in tqdm(val_dl, desc='Validation'):
            image, text_label = batch[data_key], batch['label']

            # Get the concepts to train, intersecting example's and global concepts
            concepts_to_train = self._get_concepts_to_train(batch, concepts_for_forward)

            # Forward pass
            forward_outputs = self.forward_pass(image[0], text_label[0], concepts=concepts_to_train, **forward_kwargs)
            total_loss += forward_outputs.loss # Store loss

            # Compute component accuracy
            concepts_for_component_accuracy = [c for c in concepts_to_train if c.name in component_concepts]
            if concepts_for_component_accuracy:
                component_forward_outputs = self.forward_pass(image[0], text_label[0], concepts=concepts_for_component_accuracy, **forward_kwargs)
                component_accuracy(component_forward_outputs.binary_concept_predictions, component_forward_outputs.binary_concept_labels)

            # Compute non-component accuracy
            concepts_for_non_component_accuracy = [c for c in concepts_to_train if c.name in non_component_concepts]
            if concepts_for_non_component_accuracy:
                non_component_forward_outputs = self.forward_pass(image[0], text_label[0], concepts=concepts_for_non_component_accuracy, **forward_kwargs)
                non_component_accuracy(non_component_forward_outputs.binary_concept_predictions, non_component_forward_outputs.binary_concept_labels)

        total_loss = total_loss / len(val_dl)
        component_accuracy = component_accuracy.compute().item()
        non_component_accuracy = non_component_accuracy.compute().item()

        val_output = ValidationOutput(
            loss=total_loss,
            component_accuracy=component_accuracy,
            non_component_accuracy=non_component_accuracy
        )

        return val_output

    def log(self, *args, **kwargs):
        if self.run is not None:
            self.run.log(*args, **kwargs)

    def _concepts_to_datasets(self, dataset: FeatureDataset, concepts: list[Concept] = None) -> dict[Concept, FeatureDataset]:
        '''
            For each concept in the intersection of the dataset's concepts_to_train and the global concepts,
            creates a FeatureDataset with the examples corresponding to that concept.
        '''
        # Build mapping from intersected concepts to indices
        global_concept_names = {c.name for c in concepts} if concepts else {c.name for c in self.concept_kb}
        concept_name_to_indices: dict[str, list[int]] = {}

        for i, concepts_to_train in enumerate(dataset.concepts_to_train_per_example):
            if not concepts_to_train:
                concepts_to_train = global_concept_names

            for concept_name in concepts_to_train:
                if not global_concept_names or concept_name in global_concept_names: # Intersect with global concepts
                    concept_name_to_indices.setdefault(concept_name, []).append(i)

        # Construct dataset for each concept in intersection
        concepts_to_datasets: dict[Concept, FeatureDataset] = {}

        for concept_name, indices in concept_name_to_indices.items():
            feature_paths = [dataset.feature_paths[i] for i in indices]
            labels = [dataset.labels[i] for i in indices]
            concepts_to_train_per_example = [[concept_name] for _ in indices]

            concept = self.concept_kb[concept_name]
            concept_ds = FeatureDataset(feature_paths, labels, concepts_to_train_per_example, path_to_lock=dataset.path_to_lock)
            concepts_to_datasets[concept] = concept_ds

        concepts_to_datasets = { # Topological sort via component graph
            concept : concepts_to_datasets[concept]
            for concept in self.concept_kb.in_component_order(concepts_to_datasets.keys())
        }

        return concepts_to_datasets