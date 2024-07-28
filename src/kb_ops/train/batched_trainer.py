from .base import ConceptKBTrainerBase
import torch
from model.concept import Concept
from kb_ops.dataset import FeatureDataset
from typing import  Optional
import numpy as np
from tqdm import tqdm
from kb_ops.dataset import BatchCachedFeaturesCollate
from .outputs import TrainOutput, ValidationOutput
import logging

logger = logging.getLogger(__file__)

class ConceptKBBatchedTrainerMixin(ConceptKBTrainerBase):
    def train_minimal(
        self,
        train_ds: FeatureDataset,
        n_epochs: int,
        lr: float,
        concepts: list[Concept] = None,
        batch_size: int = 64,
        dataloader_kwargs: dict = {}
    ):
        '''
            Trains the concept predictors using the given dataset one-by-one for the specified number of epochs.
            Does not perform checkpointing, validation, logging, or output collection to maximize speed.

            This training algorithm is theoretically faster than epoch-by-epoch training of all concepts as training
            concept-by-concept does not cycle between concepts' dataloaders.
            Experiments show that it is marginally faster (.3 seconds per epoch faster for ≈ 1000 example, 29 concepts dataset) than
            train_batched.

            This method has a lower memory footprint than train_batched as this does not store outputs and does not maintain
            worker processes for all concepts simultaneously.
        '''

        concept_to_train_dataset = self._concepts_to_datasets(train_ds, concepts=concepts)
        self.concept_kb.train()

        prog_bar = tqdm(concept_to_train_dataset.items())
        for concept, dataset in prog_bar:
            prog_bar.set_description(f'Training concept "{concept.name}"')

            optimizer = torch.optim.Adam(concept.predictor.parameters(), lr=lr)
            train_dl = self._get_dataloader(concept, dataset, is_train=True, batch_size=batch_size, **dataloader_kwargs)

            for epoch in range(1, n_epochs + 1):
                for batch in train_dl:
                    _ = self.batched_forward_pass(
                        batch['features'],
                        concept,
                        text_labels=batch['label'],
                        do_backward=True
                    )

                    optimizer.step()
                    optimizer.zero_grad()

    def train_batched(
        self,
        train_ds: FeatureDataset,
        val_ds: Optional[FeatureDataset],
        n_epochs: int,
        lr: float,
        concepts: list[Concept] = None,
        batch_size: int = 64,
        ckpt_every_n_epochs: int = 1,
        ckpt_dir: str = 'checkpoints',
        ckpt_fmt: str = 'concept_kb_epoch_{epoch}.pt',
        dataloader_kwargs: dict = {}
    ) -> TrainOutput:
        # For each global concept, create a dataset with its examples (consider a dataloader?)
        concept_to_train_dataset = self._concepts_to_datasets(train_ds, concepts=concepts)
        concept_to_val_dataset = self._concepts_to_datasets(val_ds, concepts=concepts) if val_ds else None
        concepts = list(concept_to_train_dataset.keys())

        train_outputs = []
        val_outputs: list[ValidationOutput] = []

        concept_to_optimizer = {c.name : torch.optim.Adam(c.predictor.parameters(), lr=lr) for c in concepts}
        concept_to_train_dl = self._concepts_to_dataloaders(concept_to_train_dataset, is_train=True, batch_size=batch_size, **dataloader_kwargs)

        for epoch in range(1, n_epochs + 1):
            logger.info(f'======== Starting Epoch {epoch}/{n_epochs} ========')
            self.concept_kb.train()

            concepts_outputs = {}
            for concept, train_dl in tqdm(concept_to_train_dl.items(), desc=f'Epoch {epoch}/{n_epochs}'):
                optimizer = concept_to_optimizer[concept.name]

                # Train concept predictor
                for batch in train_dl:
                    outputs = self.batched_forward_pass(
                        batch['features'],
                        concept,
                        text_labels=batch['label'],
                        do_backward=True
                    )
                    concepts_outputs.setdefault(concept.name, []).append(outputs)

                    optimizer.step()
                    optimizer.zero_grad()

            train_outputs.append(concepts_outputs)

            # Compute loss for logging
            concepts_to_losses = {
                concept_name : sum(output.loss for output in outputs) / len(outputs)
                for concept_name, outputs in concepts_outputs.items()
            }
            train_loss = sum(concepts_to_losses.values()) / len(concepts_to_losses)

            self.log({'train_loss': train_loss, 'epoch': epoch})

            if epoch % ckpt_every_n_epochs == 0 and ckpt_dir:
                ckpt_path = self._get_ckpt_path(ckpt_dir, ckpt_fmt, epoch)
                self.concept_kb.save(ckpt_path)
                logger.info(f'Saved checkpoint at {ckpt_path}')

            # Validate
            if concept_to_val_dataset:
                # TODO batched validation using concept_to_val_dataset
                # For now, just do single example validation
                val_dl = self._get_dataloader(val_ds, is_train=False)
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
        val_losses = [output.loss for output in val_outputs] if val_ds else None
        best_ckpt_epoch = np.argmin(val_losses) if val_losses else None
        best_ckpt_path = self._get_ckpt_path(ckpt_dir, ckpt_fmt, best_ckpt_epoch) if val_losses else None

        train_output = TrainOutput(
            best_ckpt_epoch=best_ckpt_epoch,
            best_ckpt_path=best_ckpt_path,
            train_outputs=train_outputs,
            val_outputs=val_outputs
        )

        return train_output

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

    def _concepts_to_dataloaders(
        self,
        concept_to_dataset: dict[Concept, FeatureDataset],
        is_train: bool,
        batch_size: int,
        **dataloader_kwargs
    ):
        return {
            concept : self._get_dataloader(concept, dataset, is_train=is_train, batch_size=batch_size, **dataloader_kwargs)
            for concept, dataset in concept_to_dataset.items()
        }

    def _get_dataloader(
        self,
        concept: Concept,
        dataset: FeatureDataset,
        is_train: bool,
        batch_size: int,
        **dataloader_kwargs
    ):

        default_kwargs = {
            'persistent_workers': True,
            'num_workers': 3
        }

        default_kwargs.update(dataloader_kwargs)

        component_subtree_names = [c.name for c in self.concept_kb.rooted_subtree(concept, use_component_graph=True)]
        collate_fn = BatchCachedFeaturesCollate(concept_names=component_subtree_names)
        return super()._get_dataloader(dataset, is_train=is_train, batch_size=batch_size, collate_fn=collate_fn, **default_kwargs)