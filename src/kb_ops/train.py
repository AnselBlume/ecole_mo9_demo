import os
import torch
import torch.nn.functional as F
from image_processing import LocalizeAndSegmentOutput
from torch.utils.data import DataLoader, Dataset
from model.concept import ConceptKB, Concept
from model.concept_predictor import ConceptPredictorOutput
from .feature_cache import CachedImageFeatures
from wandb.sdk.wandb_run import Run
from kb_ops.dataset import ImageDataset, list_collate, PresegmentedDataset, FeatureDataset
from .feature_pipeline import ConceptKBFeaturePipeline
from typing import Union, Optional, Any, Literal
import numpy as np
from tqdm import tqdm
from PIL.Image import Image
from torchmetrics import Accuracy
import logging

logger = logging.getLogger(__name__)

class ConceptKBExampleSampler:
    def __init__(
        self,
        concept_kb: ConceptKB,
        random_seed: int = 42
    ):
        self.concept_kb = concept_kb
        self.rng = np.random.default_rng(random_seed)

    def sample_negative_examples(
        self,
        n_pos_examples: int,
        neg_concepts: list[Concept],
        min_neg_ratio: float = 1.0
    ) -> tuple[list[Any], list[str]]:
        '''
            Samples negative examples from the given negative concepts, trying to match the given ratio.

            Arguments:
                n_pos_examples: Number of positive examples
                neg_concepts: List of negative concepts to sample at least one example of each from
                min_neg_ratio: Minimum ratio of negative examples to positive examples
                rng: Random number generator used for sampling from negative concepts

            Returns: Tuple of (sampled_examples, sampled_concept_names)
        '''
        # Decide how many negatives to sample per concept
        if min_neg_ratio < 1:
            raise ValueError('min_neg_ratio must be >= 1')

        n_neg_per_concept = int(n_pos_examples * min_neg_ratio / len(neg_concepts))

        if n_neg_per_concept < 1:
            n_neg_per_concept = 1
            logger.warning(f'Too many negative concepts to satisfy min_neg_ratio; using 1 negative example per concept')

        actual_ratio = n_neg_per_concept * len(neg_concepts) / n_pos_examples
        logger.info(f'Attempting to sample {n_neg_per_concept} negative examples per concept')

        if not rng:
            rng = np.random.default_rng()

        sampled_examples = []
        sampled_concept_names = []
        for neg_concept in neg_concepts:
            try:
                neg_examples = rng.choice(neg_concept.examples, n_neg_per_concept, replace=False)

            except ValueError: # Not enough negative examples
                logger.warning(f'Not enough examples to sample from for concept {neg_concept.name}; using all examples')
                neg_examples = neg_concept.examples

            sampled_examples.extend(neg_examples)
            sampled_concept_names.extend([neg_concept.name] * len(neg_examples))

        actual_ratio = len(sampled_examples) / n_pos_examples
        logger.info(f'Actual negative example ratio: {actual_ratio:.2f}')

        return sampled_examples, sampled_concept_names

class ConceptKBTrainer:
    UNK_LABEL = '[UNK]'

    def __init__(self, concept_kb: ConceptKB, feature_pipeline: ConceptKBFeaturePipeline, wandb_run: Run = None):

        self.concept_kb = concept_kb

        self.label_to_index: dict[str,int] = {concept.name : i for i, concept in enumerate(concept_kb)}
        self.label_to_index[self.UNK_LABEL] = -1 # For unknown labels
        self.index_to_label: dict[int,str] = {v : k for k, v in self.label_to_index.items()}

        self.feature_pipeline = feature_pipeline
        self.sampler = ConceptKBExampleSampler(concept_kb)

        self.run = wandb_run

    def _get_dataloader(self, dataset: Dataset, is_train: bool):
        return DataLoader(dataset, batch_size=1, shuffle=is_train, collate_fn=list_collate, num_workers=3, pin_memory=True)

    def _get_ckpt_path(self, ckpt_dir: str, ckpt_fmt: str, epoch: int):
        return os.path.join(ckpt_dir, ckpt_fmt.format(epoch=epoch))

    def _determine_data_key(self, dataset: Union[ImageDataset, PresegmentedDataset, FeatureDataset]):
        if isinstance(dataset, FeatureDataset):
            return 'features'

        elif isinstance(dataset, PresegmentedDataset):
            return 'segmentations'

        else:
            assert isinstance(dataset, ImageDataset)
            return 'image'

    def train(
        self,
        train_ds: Union[ImageDataset, PresegmentedDataset, FeatureDataset],
        val_ds: Optional[Union[ImageDataset, PresegmentedDataset, FeatureDataset]],
        n_epochs: int,
        lr: float,
        concepts: list[Concept] = None,
        backward_every_n_concepts: int = None,
        imgs_per_optim_step: int = 4,
        ckpt_every_n_epochs: int = 1,
        ckpt_dir: str = 'checkpoints',
        ckpt_fmt: str = 'concept_kb_epoch_{epoch}.pt'
    ):

        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

        data_key = self._determine_data_key(train_ds)
        train_dl = self._get_dataloader(train_ds, is_train=True)

        val_dl = self._get_dataloader(val_ds, is_train=False) if val_ds else None
        val_losses = [] if val_ds else None

        optimizer = torch.optim.Adam(self.concept_kb.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):
            logger.info(f'======== Starting Epoch {epoch}/{n_epochs} ========')
            self.concept_kb.train()

            for i, batch in enumerate(tqdm(train_dl, desc=f'Epoch {epoch}/{n_epochs}'), start=1):
                image_data, text_label = batch[data_key], batch['label']
                outputs = self.forward_pass(
                    image_data[0],
                    text_label[0],
                    concepts=concepts,
                    do_backward=True,
                    backward_every_n_concepts=backward_every_n_concepts
                )

                self.log({'train_loss': outputs['loss'], 'epoch': epoch})

                if i % imgs_per_optim_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            if epoch % ckpt_every_n_epochs == 0 and ckpt_dir:
                ckpt_path = self._get_ckpt_path(ckpt_dir, ckpt_fmt, epoch)
                self.concept_kb.save(ckpt_path)
                logger.info(f'Saved checkpoint at {ckpt_path}')

            # Validate
            if val_dl:
                outputs = self.validate(val_dl)
                val_losses.append(outputs['val_loss'])

                self.log({'val_loss': val_losses[-1], 'val_acc': outputs['val_acc'], 'epoch': epoch})
                logger.info(f'Validation loss: {outputs["val_loss"]}, Validation accuracy: {outputs["val_acc"]}')

        # Construct return dictionary
        best_ckpt_epoch = np.argmin(val_losses) if val_losses else None
        best_ckpt_path = self._get_ckpt_path(ckpt_dir, ckpt_fmt, best_ckpt_epoch) if val_losses else None

        ret_dict = {
            'best_ckpt_epoch': best_ckpt_epoch,
            'best_ckpt_path': best_ckpt_path
        }

        return ret_dict

    def train_concept(
        self,
        concept: Concept,
        stopping_condition: Literal['until_correct', 'n_epochs', 'validation'] = 'until_correct',
        until_correct_example_paths: list[str] = [],
        min_prob_margin: float = .1,
        n_epochs: int = 10,
        sampling_seed: int = 42
    ):
        # TODO update me
        '''
            Trains the given concept for n_epochs if provided, else until it correctly predicts the
            examples in until_correct_example_paths.

            Arguments:
                concept: Concept to train.

                stopping_condition: One of 'until_correct', 'n_epochs', or 'validation'.
                    If 'until_correct', training stops when the concept correctly predicts the examples in
                    until_correct_example_paths with a probability margin of at least min_prob_margin.
                    If 'n_epochs', training will stop after n_epochs.

                until_correct_example_paths: List of paths to example images that the concept should correctly predict if
                    stopping_condition is 'until_correct'.

                min_prob_margin: Margin of probability (computed via softmax) over other concepts that the true concept must have
                    for each example in until_correct_example_paths if stopping_condition is 'until_correct'.

                n_epochs: Number of epochs to train for if stopping_condition is 'n_epochs'.

                sampling_seed: Seed for random number generator used for sampling negative examples.
        '''
        if stopping_condition == 'validation':
            # TODO implement some way to perform validation as a stopping condition
            raise NotImplementedError('Validation is not yet implemented')

        # Train for a fixed number of epochs or until examples are predicted correctly
        else:
            # Create examples and labels
            neg_concepts = [c for c in self.concept_kb if c != concept]
            neg_examples, neg_concept_names = self.sampler.sample_negative_examples(len(concept.examples), neg_concepts)

            all_samples = concept.examples + neg_examples
            all_labels = [concept.name] * len(concept.examples) + neg_concept_names

            # Create dataset
            # TODO handle the case where some examples are presegmented and some are not with _cache...
            if all_samples[0].endswith('.pkl'):
                train_ds = PresegmentedDataset(all_samples, all_labels)
                val_ds = PresegmentedDataset(until_correct_example_paths, [concept.name] * len(until_correct_example_paths))
            else:
                train_ds = ImageDataset(all_samples, all_labels)
                val_ds = ImageDataset(until_correct_example_paths, [concept.name] * len(until_correct_example_paths))

            train_dl = self._get_dataloader(train_ds, is_train=True)
            val_dl = self._get_dataloader(val_ds, is_train=False)

            # Train for fixed number of epochs
            if stopping_condition == 'n_epochs':
                self.train(train_dl, None, n_epochs=n_epochs, lr=1e-3, concepts=[concept], ckpt_dir=None)

            else: # stopping_condition == 'until_correct'
                curr_epoch = 0
                target_concept_index = self.label_to_index[concept.name]

                while True:
                    # Train for one epoch then check the probability margins
                    curr_epoch += 1
                    self.train(train_dl, None, n_epochs=1, lr=1e-3, concepts=[concept], ckpt_dir=None)

                    predictions = self.predict(val_dl)

                    # Check probability margin of each example for stopping condition
                    all_scores = torch.stack([pred['predictors_scores'] for pred in predictions])
                    normalized_scores = F.softmax(all_scores, dim=1)
                    values, indices = normalized_scores.topk(2, dim=1) # Each of values, indices are (n_examples, 2) tensors

                    if not (indices[:, 0] == target_concept_index).all(): # Not all examples are correctly predicted
                        continue

                    prob_margins = values[:, 0] - values[:, 1] # Top prediction prob - 2nd prediction prob
                    logger.debug(f'Concept {concept.name} predicted examples with margins {prob_margins}')

                    if (prob_margins >= min_prob_margin).all():
                        break

                logger.info(f'Concept {concept.name} correctly predicted examples with margins {prob_margins} after {curr_epoch} epochs')

    @torch.inference_mode()
    def validate(self, val_dl: DataLoader):
        self.concept_kb.eval()

        total_loss = 0
        predicted_concept_outputs = []
        acc = Accuracy(task='multiclass', num_classes=len(self.concept_kb))
        data_key = self._determine_data_key(val_dl.dataset)

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
        image_data: Union[Image, LocalizeAndSegmentOutput] = None,
        unk_threshold: float = 0.,
        **forward_kwargs
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
            predicted_concept_outputs = outputs['predictors_outputs'][pred_ind].cpu()

            # If max score is less than threshold, predict UNK_LABEL
            if unk_threshold > 0 and scores[pred_ind].sigmoid() < unk_threshold:
                pred_ind = self.label_to_index[self.UNK_LABEL]

            predictions.append({
                'concept_names': outputs['concept_names'],
                'predictors_scores': scores.cpu(),
                'predicted_index': pred_ind, # This can be -1 if UNK_LABEL is predicted
                'predicted_label': self.index_to_label[pred_ind], # This can be UNK_LABEL
                'predicted_concept_outputs': predicted_concept_outputs, # This will always be maximizing concept
                'true_index': true_ind if predict_dl is not None else None,
                'true_concept_outputs': None if predict_dl is None or true_ind < 0 else outputs['predictors_outputs'][true_ind].cpu()
            })

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

    def forward_pass(
        self,
        image_data: Union[Image, LocalizeAndSegmentOutput, CachedImageFeatures],
        text_label: str = None,
        concepts: list[Concept] = None,
        do_backward: bool = False,
        backward_every_n_concepts: int = None,
        return_segmentations: bool = False,
        return_trained_attr_scores: bool = False,
        return_features: bool = False
    ):

        if isinstance(image_data, CachedImageFeatures):
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
        all_features = []

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

            if return_features:
                all_features.append(features)

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

        # Return results
        ret_dict = {
            'loss': total_loss if text_label is not None else None,
            'predictors_outputs': outputs,
            'concept_names': [concept.name for concept in concepts]
        }

        if return_segmentations:
            ret_dict['segmentations'] = segmentations

        if return_trained_attr_scores:
            ret_dict['trained_attr_scores'] = cached_trained_attr_scores.cpu()

        if return_features:
            ret_dict['all_concept_features'] = all_features

        return ret_dict

    def log(self, *args, **kwargs):
        if self.run is not None:
            self.run.log(*args, **kwargs)