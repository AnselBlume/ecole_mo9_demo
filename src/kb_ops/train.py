import os
import torch
import torch.nn.functional as F
from controller import Controller
from torch.utils.data import DataLoader
from model.concept import ConceptKB
from model.concept_predictor import ConceptPredictorOutput
from model.features import ImageFeatures
from wandb.sdk.wandb_run import Run
from kb_ops.dataset import ImageDataset, list_collate, PresegmentedDataset
from feature_extraction import FeatureExtractor
from typing import Union
from tqdm import tqdm
from PIL.Image import Image
from torchmetrics import Accuracy
import logging

logger = logging.getLogger(__name__)

class ConceptKBTrainer:
    def __init__(
        self,
        concept_kb: ConceptKB,
        controller: Controller,
        feature_extractor: FeatureExtractor,
        wandb_run: Run = None
    ):

        self.concept_kb = concept_kb
        self.label_to_index = {concept.name : torch.tensor(i, dtype=torch.float32) for i, concept in enumerate(concept_kb)}
        self.controller = controller
        self.feature_extractor = feature_extractor
        self.run = wandb_run

    def train(
        self,
        train_ds: Union[ImageDataset, PresegmentedDataset],
        val_ds: Union[ImageDataset, PresegmentedDataset],
        n_epochs: int,
        lr: float,
        backward_every_n_concepts: int = None,
        imgs_per_optim_step: int = 4,
        ckpt_every_n_epochs: int = 1,
        ckpt_dir: str = 'checkpoints'
    ):

        os.makedirs(ckpt_dir, exist_ok=True)

        is_presegmented = isinstance(train_ds, PresegmentedDataset)
        data_key = 'segmentations' if is_presegmented else 'image'

        train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=list_collate)
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=list_collate)
        optimizer = torch.optim.Adam(self.concept_kb.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):
            logger.info(f'======== Starting Epoch {epoch}/{n_epochs} ========')

            for i, batch in enumerate(tqdm(train_dl, desc=f'Epoch {epoch}/{n_epochs}'), start=1):
                image_data, label = batch[data_key], batch['label']
                outputs = self.forward_pass(image_data[0], self.label_to_index[label[0]], backward_every_n_concepts=backward_every_n_concepts)

                self.log({'train_loss': outputs['loss'], 'epoch': epoch})

                if i % imgs_per_optim_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            if epoch % ckpt_every_n_epochs == 0:
                ckpt_path = os.path.join(ckpt_dir, f'concept_kb_epoch_{epoch}.pt')
                self.concept_kb.save(ckpt_path)
                logger.info(f'Saved checkpoint at {ckpt_path}')

            # Validate
            outputs = self.validate(val_dl)

            self.log({'val_loss': outputs['val_loss'], 'val_acc': outputs['val_acc'], 'epoch': epoch})
            logger.info(f'Validation loss: {outputs["val_loss"]}, Validation accuracy: {outputs["val_acc"]}')

        return self.concept_kb

    @torch.inference_mode()
    def validate(self, val_dl: DataLoader):
        total_loss = 0
        predicted_concept_outputs = []
        acc = Accuracy(task='multiclass', num_classes=len(self.concept_kb))

        for i, batch in enumerate(tqdm(val_dl, desc='Validation'), start=1):
            image, label = batch['image'], batch['label']
            outputs = self.forward_pass(image[0], label[0], do_backward=False)

            total_loss += outputs['loss'] # Store loss

            # Compute predictions and accuracy
            scores = torch.tensor([output.cum_score for output in outputs['predictors_outputs']])
            pred_ind = scores.argmax(dim=0, keepdim=True) # (1,) IntTensor
            true_ind = self.label_to_index[label].int()

            acc(pred_ind, true_ind)

            # Store predicted output
            predicted_concept_outputs.append(outputs['predictors_outputs'][pred_ind.item()])

        total_loss = total_loss / len(val_dl)
        val_acc = acc.compute().item()

        return {
            'val_loss': total_loss,
            'val_acc': val_acc,
            'predicted_concept_outputs': predicted_concept_outputs
        }

    def forward_pass(
        self,
        image_data: Union[Image, dict],
        label: str,
        do_backward: bool = True,
        backward_every_n_concepts: int = None
    ):
        # TODO Make an actual Segmentations data type and have localize_and_segment return it

        # Get region crops
        if isinstance(image_data, Image):
            segmentations = self.controller.localize_and_segment(
                image=image_data,
                concept_name='',
                concept_parts=[],
                remove_background=True
            )
            image = image_data

        else: # Assume they're preprocessed segmentations
            segmentations = image_data
            image = segmentations['image']

        region_crops = segmentations['part_crops']
        if region_crops == []:
            region_crops = [image]

        # Get all concept predictions
        total_loss = 0
        curr_loss = 0
        outputs = []

        for i, concept in enumerate(self.concept_kb, start=1):
            zs_attrs = [attr.query for attr in concept.zs_attributes]

            with torch.no_grad():
                features: ImageFeatures = self.feature_extractor(image, region_crops, zs_attrs)

            output: ConceptPredictorOutput = concept.predictor(features)
            concept_loss = F.binary_cross_entropy_with_logits(output.cum_score, label.to(output.cum_score)) / len(self.concept_kb)

            curr_loss += concept_loss
            total_loss += concept_loss.item()
            outputs.append(output.to('cpu'))

            if (
                do_backward and backward_every_n_concepts is not None
                and (i % backward_every_n_concepts == 0 or i == len(self.concept_kb))
            ):
                curr_loss.backward()
                curr_loss = 0

        if do_backward and backward_every_n_concepts is None: # Backward if we weren't doing it every K concepts
            curr_loss.backward()

        return {
            'loss': total_loss,
            'predictors_outputs': outputs
        }

    def log(self, *args, **kwargs):
        if self.run is not None:
            self.run.log(*args, **kwargs)