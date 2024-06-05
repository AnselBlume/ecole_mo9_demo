import os
import torch
from model.concept import ConceptKB, Concept, ConceptExample
from wandb.sdk.wandb_run import Run
from dataclasses import dataclass, field
from .dataset import ImageDataset, PresegmentedDataset, FeatureDataset, extend_with_global_negatives
from .feature_pipeline import ConceptKBFeaturePipeline
from typing import Union, Optional, Any, Literal, Callable
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import numpy as np
from tqdm import tqdm
import logging
from .forward import ConceptKBForwardBase, ForwardOutput, DictDataClass
from.example_sampler import ConceptKBExampleSampler

logger = logging.getLogger(__name__)

@dataclass
class ValidationOutput(DictDataClass):
    loss: float
    component_accuracy: float
    non_component_accuracy: float

@dataclass
class TrainOutput(DictDataClass):
    best_ckpt_epoch: int = None
    best_ckpt_path: str = None
    train_outputs: list[ForwardOutput] = None
    val_outputs: Optional[list[ValidationOutput]] = field(
        default=None,
        metadata={'description': 'List of outputs from validation dataloader if val_dl is provided'}
    )

class ConceptKBTrainer(ConceptKBForwardBase):
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

    def train(
        self,
        train_ds: Union[ImageDataset, PresegmentedDataset, FeatureDataset],
        val_ds: Optional[Union[ImageDataset, PresegmentedDataset, FeatureDataset]],
        n_epochs: int,
        lr: float,
        concepts: list[Concept] = None,
        leaf_nodes_only: bool = False,
        backward_every_n_concepts: int = None,
        imgs_per_optim_step: int = 4,
        ckpt_every_n_epochs: int = 1,
        ckpt_dir: str = 'checkpoints',
        ckpt_fmt: str = 'concept_kb_epoch_{epoch}.pt',
        set_score_to_zero_at_indices: list[int] = []
    ) -> TrainOutput:

        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

        # Prepare concepts
        if concepts is None:
            if leaf_nodes_only:
                logger.info('Training leaf concepts only')
                concepts = self.concept_kb.leaf_concepts

            else:
                logger.info('Training all concepts in ConceptKB, including internal nodes')
                concepts = self.concept_kb

        concepts = {c.name : c for c in concepts}
        data_key = self._determine_data_key(train_ds)

        train_dl = self._get_dataloader(train_ds, is_train=True)
        train_outputs = []

        val_dl = self._get_dataloader(val_ds, is_train=False) if val_ds else None
        val_outputs: list[ValidationOutput] = []

        optimizer = torch.optim.Adam(self.concept_kb.parameters(), lr=lr)
        set_score_to_zero_at_indices = set(set_score_to_zero_at_indices)

        for epoch in range(1, n_epochs + 1):
            logger.info(f'======== Starting Epoch {epoch}/{n_epochs} ========')
            self.concept_kb.train()

            for i, batch in enumerate(tqdm(train_dl, desc=f'Epoch {epoch}/{n_epochs}'), start=1):
                image_data, text_label = batch[data_key], batch['label']
                index = batch['index'][0] # To check whether to set score to zero

                # Get the concepts to train, intersecting example's and global concepts
                concepts_to_train = self._get_concepts_to_train(batch, concepts)

                if not concepts_to_train:
                    logger.warning(f'No concepts to train after intersection with global concepts for example {index}; skipping')
                    continue

                # Forward pass
                outputs = self.forward_pass(
                    image_data[0],
                    text_label[0],
                    concepts=concepts_to_train,
                    do_backward=True,
                    backward_every_n_concepts=backward_every_n_concepts,
                    set_score_to_zero=set_score_to_zero_at_indices and index in set_score_to_zero_at_indices
                )
                train_outputs.append(outputs)

                self.log({'train_loss': outputs['loss'], 'epoch': epoch})

                if i % imgs_per_optim_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            optimizer.step()
            optimizer.zero_grad()

            if epoch % ckpt_every_n_epochs == 0 and ckpt_dir:
                ckpt_path = self._get_ckpt_path(ckpt_dir, ckpt_fmt, epoch)
                self.concept_kb.save(ckpt_path)
                logger.info(f'Saved checkpoint at {ckpt_path}')

            # Validate
            if val_dl:
                outputs = self.validate(val_dl)
                val_outputs.append(outputs)

                self.log({
                    'val_loss': outputs.loss,
                    'val_component_acc': outputs.component_accuracy,
                    'val_non_component_acc': outputs.non_component_accuracy,
                    'epoch': epoch
                })

                logger.info(
                    f'Validation loss: {outputs.loss},'
                    + f' Validation component accuracy: {outputs.component_accuracy:.4f},'
                    + f' Validation non-component accuracy: {outputs.non_component_accuracy:.4f}'
                )

        # Construct return dictionary
        val_losses = [output.loss for output in val_outputs] if val_dl else None
        best_ckpt_epoch = np.argmin(val_losses) if val_losses else None
        best_ckpt_path = self._get_ckpt_path(ckpt_dir, ckpt_fmt, best_ckpt_epoch) if val_losses else None

        train_output = TrainOutput(
            best_ckpt_epoch=best_ckpt_epoch,
            best_ckpt_path=best_ckpt_path,
            train_outputs=train_outputs,
            val_outputs=val_outputs
        )

        return train_output

    def train_concept(
        self,
        concept: Concept,
        *, # Force the use of kwargs after this point due to API changes
        stopping_condition: Literal['n_epochs', 'validation'] = 'n_epochs',
        n_epochs: int = 10,
        use_descendants_as_positives: bool = True,
        n_sampled_positives_per_descendant: int = 3,
        use_concepts_as_negatives: bool = False,
        sample_all_negatives: bool = False,
        sample_only_siblings_for_negatives: bool = True,
        sample_only_leaf_nodes_for_negatives: bool = False,
        post_sampling_hook: Callable[[list[ConceptExample]], Any] = None,
        n_global_negatives: int = 250,
        **train_kwargs
    ) -> TrainOutput:
        '''
            Trains the given concept for n_epochs if provided, else until it correctly predicts the
            examples in until_correct_example_paths.

            Arguments:
                concept: Concept to train.

                stopping_condition: One of 'n_epochs' or 'validation'.
                    If 'n_epochs', training will stop after n_epochs.

                n_epochs: Number of epochs to train for if stopping_condition is 'n_epochs'.

                use_descendants_as_positives: Whether to use descendants of the concept as positive examples.
                n_sampled_positives_per_descendant: Number of positive examples to sample per descendant.

                use_concepts_as_negatives: Whether to use negative examples from other concepts in the ConceptKB.

                n_global_negatives: Number of global negative examples to sample for training.

        '''
        if not concept.examples:
            raise ValueError('Concept must have examples to train')

        if stopping_condition == 'validation':
            # Implement some way to perform validation as a stopping condition
            raise NotImplementedError('Validation is not yet implemented')

        # Train for a fixed number of epochs or until examples are predicted correctly
        else:
            # Create examples and labels

            # Potentially sample negative concepts
            if use_concepts_as_negatives:
                if sample_only_siblings_for_negatives:
                    siblings = {} # Not including self
                    for parent in concept.parent_concepts.values():
                        for child in parent.child_concepts.values():
                            if child.name not in siblings and child.name != concept.name:
                                siblings[child.name] = child

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
                    neg_examples, neg_concept_names = self.sampler.sample_negative_examples(len(concept.examples), neg_concepts)

            else:
                neg_examples = []
                neg_concept_names = []

            # Construct positive examples
            pos_examples = concept.examples
            concept_names = [concept.name] * len(pos_examples)

            if use_descendants_as_positives:
                descendants = self.concept_kb.rooted_subtree(concept)
                descendants = [c for c in descendants if c.name != concept.name] # Exclude self

                descendant_pos_examples, descendant_concept_names = self.sampler.sample_examples(descendants, n_examples_per_concept=n_sampled_positives_per_descendant)
                pos_examples.extend(descendant_pos_examples)
                concept_names.extend(descendant_concept_names)

            pos_labels = [ # Handle concept-specific negatives
                concept_name if not ex.is_negative else FeatureDataset.NEGATIVE_LABEL
                for ex, concept_name in zip(pos_examples, concept_names)
            ]

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

            if post_sampling_hook:
                post_sampling_hook(all_samples + global_negatives)

            extend_with_global_negatives(train_ds, global_negatives)

            # Train for fixed number of epochs
            train_kwargs = train_kwargs.copy()
            train_kwargs.update({
                'ckpt_dir': None,
                'concepts': [concept],
                'lr': train_kwargs.get('lr', 1e-2)
            })

            if stopping_condition == 'n_epochs':
                results = self.train(train_ds, None, n_epochs=n_epochs, **train_kwargs)

            else: # stopping_condition == 'until_correct'
                raise ValueError('until_correct is disabled as images may not be linearly separable, resulting in an infinite loop')

            return results

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