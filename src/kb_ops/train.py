import os
import torch
import torch.nn.functional as F
from model.concept import ConceptKB, Concept, ConceptExample
from wandb.sdk.wandb_run import Run
from .predict import ConceptKBPredictor
from .dataset import ImageDataset, PresegmentedDataset, FeatureDataset
from .feature_pipeline import ConceptKBFeaturePipeline
from typing import Union, Optional, Any, Literal, Callable
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import numpy as np
from tqdm import tqdm
import logging
from .forward import ConceptKBForwardBase

logger = logging.getLogger(__name__)

class ConceptKBExampleSampler:
    def __init__(
        self,
        concept_kb: ConceptKB,
        random_seed: int = 42
    ):
        self.concept_kb = concept_kb
        self.rng = np.random.default_rng(random_seed)

    def get_all_examples(self, concepts: list[Concept]) -> tuple[list[ConceptExample], list[str]]:
        all_examples = []
        all_labels = []
        for concept in concepts:
            n_examples = len(concept.examples)
            all_examples.extend(concept.examples)
            all_labels.extend([concept.name] * n_examples)

        return all_examples, all_labels

    def sample_negative_examples(
        self,
        n_pos_examples: int,
        neg_concepts: list[Concept],
        min_neg_ratio_per_concept: float = 1.0
    ) -> tuple[list[ConceptExample], list[str]]:
        '''
            Samples negative examples from the given negative concepts, trying to match the given ratio.

            Arguments:
                n_pos_examples: Number of positive examples
                neg_concepts: List of negative concepts to sample at least one example of each from
                min_neg_ratio: Minimum ratio of negative examples to positive examples
                rng: Random number generator used for sampling from negative concepts

            Returns: Tuple of (sampled_examples, sampled_concept_names)
        '''
        # TODO Sample descendants of each negative concept if the concept is not a leaf node

        # Decide how many negatives to sample per concept
        n_neg_per_concept = max(int(min_neg_ratio_per_concept * n_pos_examples), 1)

        logger.info(f'Attempting to sample {n_neg_per_concept} negative examples per concept')

        sampled_examples = []
        sampled_concept_names = []
        for neg_concept in neg_concepts:
            try:
                neg_examples = self.rng.choice(neg_concept.examples, n_neg_per_concept, replace=False)

            except ValueError: # Not enough negative examples
                logger.warning(f'Not enough examples to sample from for concept {neg_concept.name}; using all examples')
                neg_examples = neg_concept.examples

            sampled_examples.extend(neg_examples)
            sampled_concept_names.extend([neg_concept.name] * len(neg_examples))

        return sampled_examples, sampled_concept_names

class ConceptKBTrainer(ConceptKBForwardBase):
    def __init__(self, concept_kb: ConceptKB, feature_pipeline: ConceptKBFeaturePipeline = None, wandb_run: Run = None):
        '''
            feature_pipeline must be provided if not using a FeatureDataset.
        '''
        super().__init__(concept_kb, feature_pipeline)

        self.sampler = ConceptKBExampleSampler(concept_kb)
        self.run = wandb_run

    def _get_ckpt_path(self, ckpt_dir: str, ckpt_fmt: str, epoch: int):
        return os.path.join(ckpt_dir, ckpt_fmt.format(epoch=epoch))

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
    ) -> dict[str, Any]:

        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

        # Prepare concepts
        if concepts is None:
            if leaf_nodes_only:
                logger.info('Training leaf concepts only')
                concepts = self.concept_kb.leaf_concepts

            else:
                logger.info('Training all concepts in ConceptKB, including internal nodes')
                concepts = list(self.concept_kb)

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
        until_correct_examples: list[ConceptExample] = [],
        min_prob_margin: float = .1,
        n_epochs_between_predictions: int = 5,
        sample_all_negatives: bool = False,
        sample_only_siblings: bool = False,
        sample_only_leaf_nodes: bool = True,
        post_sampling_hook: Callable[[list[ConceptExample]], Any] = None,
        n_epochs: int = 10,
        **train_kwargs
    ) -> dict[str, Any]:
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
            # Implement some way to perform validation as a stopping condition
            raise NotImplementedError('Validation is not yet implemented')

        # Train for a fixed number of epochs or until examples are predicted correctly
        else:
            # Create examples and labels

            # Sample the negative concepts
            if sample_only_siblings:
                neg_concepts = [
                    child
                    for parent in concept.parent_concepts.values()
                    for child in parent.child_concepts.values()
                    if child != concept
                ]
            elif sample_only_leaf_nodes:
                neg_concepts = [c for c in self.concept_kb.leaf_concepts if c != concept]
            else:
                neg_concepts = [c for c in self.concept_kb if c != concept]

            # Sample the negative examples from the negative concepts
            if sample_all_negatives:
                neg_examples, neg_concept_names = self.sampler.get_all_examples(neg_concepts)
            else:
                neg_examples, neg_concept_names = self.sampler.sample_negative_examples(len(concept.examples), neg_concepts)

            all_samples = concept.examples + neg_examples
            all_labels = [concept.name] * len(concept.examples) + neg_concept_names

            if post_sampling_hook:
                post_sampling_hook(all_samples)

            # Assume features are already cached and get paths
            all_feature_paths = [sample.image_features_path for sample in all_samples]
            if any(feature_path is None for feature_path in all_feature_paths):
                raise RuntimeError('All examples must have image_features_path set to train individual Concept')

            until_correct_example_paths = [ex.image_features_path for ex in until_correct_examples]

            # Create dataset
            train_ds = FeatureDataset(all_feature_paths, all_labels)
            val_ds = FeatureDataset(until_correct_example_paths, [concept.name] * len(until_correct_example_paths))

            val_dl = self._get_dataloader(val_ds, is_train=False)

            # Train for fixed number of epochs
            train_kwargs = train_kwargs.copy()
            train_kwargs.update({
                'ckpt_dir': None,
                'concepts': [concept],
                'lr': train_kwargs.get('lr', 1e-2),
            })

            if stopping_condition == 'n_epochs':
                results = self.train(train_ds, None, n_epochs=n_epochs, **train_kwargs)

            else: # stopping_condition == 'until_correct'
                predictor = ConceptKBPredictor(self.concept_kb, self.feature_pipeline)
                curr_epoch = 0
                target_concept_index = predictor.leaf_name_to_leaf_ind[concept.name]

                while True:
                    # Train for one epoch then check the probability margins
                    curr_epoch += 1
                    results = self.train(train_ds, None, n_epochs=n_epochs_between_predictions, **train_kwargs)

                    predictions = predictor.predict(val_dl)

                    # Check probability margin of each example for stopping condition
                    all_scores = torch.stack([pred['predictors_scores'] for pred in predictions])
                    normalized_scores = F.softmax(all_scores, dim=1)
                    logger.debug(f'Normalized scores: {normalized_scores}')

                    values, indices = normalized_scores.topk(2, dim=1) # Each of values, indices are (n_examples, 2) tensors

                    logger.debug(f'Top two scores: {values}')
                    logger.debug(f'Target scores: {normalized_scores[:, target_concept_index]}')

                    if not (indices[:, 0] == target_concept_index).all(): # Not all examples are correctly predicted
                        continue

                    prob_margins = values[:, 0] - values[:, 1] # Top prediction prob - 2nd prediction prob
                    logger.debug(f'Concept {concept.name} predicted examples with margins {prob_margins}')

                    if (prob_margins >= min_prob_margin).all():
                        break

                logger.info(f'Concept {concept.name} correctly predicted examples with margins {prob_margins} after {curr_epoch} loops')

            return results

    @torch.inference_mode()
    def validate(self, val_dl: DataLoader, leaf_nodes_only_for_accuracy=True, **forward_kwargs):
        self.concept_kb.eval()

        total_loss = 0
        predicted_concept_outputs = []
        acc = Accuracy(task='multiclass', num_classes=len(self.concept_kb))
        data_key = self._determine_data_key(val_dl.dataset)

        if leaf_nodes_only_for_accuracy:
            forward_kwargs['concepts'] = self.concept_kb.leaf_concepts

        for batch in tqdm(val_dl, desc='Validation'):
            image, text_label = batch[data_key], batch['label']
            outputs = self.forward_pass(image[0], text_label[0], **forward_kwargs)

            total_loss += outputs['loss'] # Store loss

            # Compute predictions and accuracy
            scores = torch.tensor([output.cum_score for output in outputs['predictors_outputs']])

            pred_ind = scores.argmax(dim=0, keepdim=True) # (1,) IntTensor

            # Compute true index
            true_ind = self.label_to_ind[text_label[0]] # Global index

            if leaf_nodes_only_for_accuracy: # To leaf index
                true_ind = self.global_ind_to_leaf_ind[true_ind]

            true_ind = torch.tensor(true_ind).unsqueeze(0) # (1,)

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

    def log(self, *args, **kwargs):
        if self.run is not None:
            self.run.log(*args, **kwargs)