import os
import torch
import torch.nn.functional as F
from image_processing import LocalizerAndSegmenter
from torch.utils.data import DataLoader, Dataset
from model.concept import ConceptKB, Concept
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
    UNK_LABEL = '[UNK]'

    def __init__(
        self,
        concept_kb: ConceptKB,
        feature_extractor: FeatureExtractor,
        loc_and_seg: LocalizerAndSegmenter = None,
        wandb_run: Run = None
    ):

        self.concept_kb = concept_kb

        self.label_to_index: dict[str,int] = {concept.name : i for i, concept in enumerate(concept_kb)}
        self.label_to_index[self.UNK_LABEL] = -1 # For unknown labels
        self.index_to_label: dict[int,str] = {v : k for k, v in self.label_to_index.items()}

        self.loc_and_seg = loc_and_seg
        self.feature_extractor = feature_extractor
        self.run = wandb_run

    def get_dataloader(self, dataset: Dataset, is_train: bool):
        return DataLoader(dataset, batch_size=1, shuffle=is_train, collate_fn=list_collate, num_workers=3, pin_memory=True)

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

        train_dl = self.get_dataloader(train_ds, is_train=True)
        val_dl = self.get_dataloader(val_ds, is_train=False)
        optimizer = torch.optim.Adam(self.concept_kb.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):
            logger.info(f'======== Starting Epoch {epoch}/{n_epochs} ========')
            self.concept_kb.train()

            for i, batch in enumerate(tqdm(train_dl, desc=f'Epoch {epoch}/{n_epochs}'), start=1):
                image_data, text_label = batch[data_key], batch['label']
                outputs = self.forward_pass(
                    image_data[0],
                    text_label[0],
                    do_backward=True,
                    backward_every_n_concepts=backward_every_n_concepts
                )

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
        self.concept_kb.eval()

        total_loss = 0
        predicted_concept_outputs = []
        acc = Accuracy(task='multiclass', num_classes=len(self.concept_kb))
        data_key = 'segmentations' if isinstance(val_dl.dataset, PresegmentedDataset) else 'image'

        for batch in tqdm(val_dl, desc='Validation'):
            image, text_label = batch[data_key], batch['label']
            outputs = self.forward_pass(image[0], text_label[0])

            total_loss += outputs['loss'] # Store loss

            # Compute predictions and accuracy
            scores = torch.tensor([output.cum_score for output in outputs['predictors_outputs']])
            pred_ind = scores.argmax(dim=0, keepdim=True) # (1,) IntTensor
            true_ind = torch.tensor(self.label_to_index[text_label[0]]).unsqueeze(0) # (1,)

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

    @torch.inference_mode()
    def predict(
        self,
        predict_dl: DataLoader = None,
        image_data: Union[Image, dict] = None,
        unk_threshold: float = 0.
    ) -> Union[list[dict], dict]:
        '''
            unk_threshold: Number between [0,1]. If sigmoid(max concept score) is less than this,
                outputs self.label_to_index[self.UNK_LABEL] as the predicted label.

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

            # If max score is less than threshold, predict UNK_LABEL
            if unk_threshold > 0 and scores[pred_ind].sigmoid() < unk_threshold:
                pred_ind = self.label_to_index[self.UNK_LABEL]

            predictions.append({
                'concept_names': outputs['concept_names'],
                'predictors_scores': scores.cpu(),
                'predicted_index': pred_ind,
                'predicted_label': self.index_to_label[pred_ind],
                'predicted_concept_outputs': outputs['predictors_outputs'][pred_ind].cpu(),
                'true_index': true_ind if predict_dl is not None else None,
                'true_concept_outputs': None if predict_dl is None or true_ind < 0 else outputs['predictors_outputs'][true_ind].cpu()
            })

        if predict_dl is not None:
            data_key = 'segmentations' if isinstance(predict_dl.dataset, PresegmentedDataset) else 'image'

            for batch in tqdm(predict_dl, desc='Prediction'):
                image, text_label = batch[data_key], batch['label']
                outputs = self.forward_pass(image[0], text_label[0])
                process_outputs(outputs)

            return predictions

        else: # image_data is not None
            outputs = self.forward_pass(image_data)
            process_outputs(outputs)

            return predictions[0]

    def forward_pass(
        self,
        image_data: Union[Image, dict],
        text_label: str = None,
        concepts: list[Concept] = None,
        do_backward: bool = False,
        backward_every_n_concepts: int = None
    ):
        # TODO Make an actual Segmentations data type and have localize_and_segment return it

        # Get region crops
        if isinstance(image_data, Image):
            if self.loc_and_seg is None:
                raise ValueError('LocalizerAndSegmenter is required for online localization and segmentation')

            segmentations = self.loc_and_seg.localize_and_segment(
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

        # Cache to avoid recomputation for each image
        visual_features = None
        trained_attr_scores = None

        concepts = concepts if concepts is not None else self.concept_kb
        for i, concept in enumerate(concepts, start=1):
            zs_attrs = [attr.query for attr in concept.zs_attributes]

            # Compute image features
            with torch.no_grad():
                features: ImageFeatures = self.feature_extractor(
                    image,
                    region_crops,
                    zs_attrs,
                    cached_visual_features=visual_features,
                    cached_trained_attr_scores=trained_attr_scores
                )

            # Cache visual features and trained attribute scores
            if visual_features is None:
                visual_features = torch.cat([features.image_features, features.region_features], dim=0)

            if trained_attr_scores is None:
                trained_attr_scores = torch.cat([features.trained_attr_img_scores, features.trained_attr_region_scores], dim=0)

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
                and (i % backward_every_n_concepts == 0 or i == len(self.concept_kb))
            ):
                curr_loss.backward()
                curr_loss = 0

        if do_backward and backward_every_n_concepts is None: # Backward if we weren't doing it every K concepts
            curr_loss.backward()

        return {
            'loss': total_loss if text_label is not None else None,
            'predictors_outputs': outputs,
            'concept_names': [concept.name for concept in concepts]
        }

    def log(self, *args, **kwargs):
        if self.run is not None:
            self.run.log(*args, **kwargs)