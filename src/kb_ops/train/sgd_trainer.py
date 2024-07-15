from .base import ConceptKBTrainerBase
import os
import torch
from model.concept import  Concept
from kb_ops.dataset import ImageDataset, PresegmentedDataset, FeatureDataset
from typing import Union, Optional
import numpy as np
from tqdm import tqdm
from .outputs import TrainOutput, ValidationOutput
import logging

logger = logging.getLogger(__file__)

class ConceptKBSGDTrainerMixin(ConceptKBTrainerBase):
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
        ckpt_fmt: str = 'concept_kb_epoch_{epoch}.pt'
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

        for epoch in range(1, n_epochs + 1):
            logger.info(f'======== Starting Epoch {epoch}/{n_epochs} ========')
            self.concept_kb.train()

            for i, batch in enumerate(tqdm(train_dl, desc=f'Epoch {epoch}/{n_epochs}'), start=1):
                image_data, text_label = batch[data_key], batch['label']
                index = batch['index'][0]

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
                    backward_every_n_concepts=backward_every_n_concepts
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
