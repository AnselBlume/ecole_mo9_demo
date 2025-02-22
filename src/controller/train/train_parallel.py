import torch
from .base import ControllerTrainMixinBase
from model.concept import Concept, ConceptExample
from kb_ops.train import ConceptKBTrainer
from kb_ops.feature_pipeline import ConceptKBFeaturePipeline
import time
from typing import Union
from kb_ops.dataset import FeatureDataset
from kb_ops.concurrency import (
    ConcurrentTrainingConceptSelector,
    LockType,
    get_lock_generator
)
import traceback
import sys
import logging

logger = logging.getLogger(__file__)

# NOTE Cannot use NFS with mp as otherwise get "device or resource busy" errors
# This code should be set in the main file before any other imports, or in the shell
# tmp_dir = '/scratch/tmp'
# os.makedirs(tmp_dir, exist_ok=True)
# os.environ['TMPDIR'] = tmp_dir

import multiprocessing as mp # Must be imported after setting TMPDIR to change default temp dir
mp.set_start_method('spawn', force=True) # Necessary for CUDA to work in spawned processes

def _launch_training(
    trainer: ConceptKBTrainer,
    concept_name: str,
    examples: list[ConceptExample],
    dataset: FeatureDataset,
    return_queue: mp.Queue,
    n_epochs: int = 5,
    device: Union[str, torch.device] = 'cpu',
    **train_concept_kwargs,
):
    # TODO allow the spawned process to construct its own dataset, recache its own scores by receiving
    # Cacher, path_to_file_locks

    # Note that without special handling these log messages go into the void as each process
    # has a different stdout
    try:
        logger.info(f'Training concept: {concept_name}\n\n')
        trainer.concept_kb.to(device) # Move entire ConceptKB since want components on CUDA as well
        concept = trainer.concept_kb.get_concept(concept_name)

        # Train for fixed number of epochs
        trainer.train_concept(
            concept_name,
            samples_and_dataset=(examples, dataset),
            n_epochs=n_epochs,
            **train_concept_kwargs,
        )
    except Exception as e:
        logger.info("+++++++++++++++++++++++++++++++++++++++++++++++")
        traceback.print_exc()
        logger.info(sys.exc_info())
        logger.info(f"Error training concepts: {e}")
        logger.info("concept device", concept.predictor.device)
        logger.info("+++++++++++++++++++++++++++++++++++++++++++++++")

        raise e

    logger.info(f"Training complete for concept: {concept.name}")

    # Return all to CPU to cleanly terminate process and return concept to main process
    trainer.concept_kb.cpu()
    return_queue.put((concept, device))

class ControllerTrainParallelMixin(ControllerTrainMixinBase):
    def train_concepts_parallel(
        self,
        concept_names: list[str],
        n_epochs: int = 5,
        use_concepts_as_negatives: bool = True,
        max_retrieval_distance=.01,
        concepts_to_train_kwargs: dict = {},
        devices: Union[list[str], list[int], list[torch.device]] = None, # None: use all
        lock_type: LockType = LockType.FILE_LOCK,
        **train_concept_kwargs
    ):
        if devices:
            if isinstance(devices[0], str):
                devices = [torch.device(d) for d in devices]
            elif isinstance(devices[0], int):
                devices = [torch.device(f'cuda:{d}') for d in devices]
            else: # Assume torch.device
                pass
        else:
            devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]

        available_devices = devices

        # Ensure features are prepared, only generating those which don't already exist or are dirty
        # Cache all concepts, since we might sample from concepts whose examples haven't been cached yet
        self.cacher.cache_segmentations(only_uncached_or_dirty=True)
        self.cacher.cache_features(only_uncached_or_dirty=True)

        orig_devices = {concept.name : concept.predictor.device for concept in self.concept_kb}
        self.concept_kb.cpu() # Can't be on CUDA when passing to other processes

        concepts = [
            self.retrieve_concept(c, max_retrieval_distance=max_retrieval_distance)
            for c in concept_names
        ]

        concepts_to_train: dict[Concept, None] = {}
        for concept in concepts:
            concepts_to_train.update(
                dict.fromkeys(
                    self._get_concepts_to_train_to_update_concept(
                        concept, **concepts_to_train_kwargs
                    )
                )
            )

        logger.info(f'Concepts to train: {[c.name for c in concepts_to_train]}')

        concept_selector = ConcurrentTrainingConceptSelector(list(concepts_to_train))
        all_processes: list[mp.Process] = []

        feature_pipeline = ConceptKBFeaturePipeline(None, None, config=self.feature_pipeline.config) # Pass config along
        trainer = ConceptKBTrainer(self.concept_kb, feature_pipeline)

        # Manager.Queue instead of mp.Queue seems to avoid FileNotFoundError in Queue.get() for lots of processes
        manager = mp.Manager()
        return_queue = manager.Queue()

        lock_generator = get_lock_generator(lock_type)

        start_time = time.time()
        while not concept_selector.is_training_complete:
            if len(available_devices) == 0 or not concept_selector.has_concept_available():
                # Wait for a process to complete
                trained_concept, device = return_queue.get() # Blocks until a process completes
                trained_concept: Concept
                device: torch.device

                # Probably doesn't hurt to copy the pickled concept (with its pickled child/component objects) back into the main process' ConceptKB,
                # but safer to just set the predictor's state_dict and use the main process' version of the concept to avoid potential problems
                concept = self.concept_kb[trained_concept.name]
                concept.predictor.load_state_dict(trained_concept.predictor.state_dict()) # This is on the CPU

                available_devices.append(device)
                concept_selector.mark_concept_completed(concept)

                logger.info(f'Concept training complete: {concept.name}; {concept_selector.num_concepts_incomplete} concept(s) remaining')

                if concept_selector.is_training_complete: # This was the last concept
                    break

                # We may still be waiting on a dependency for the next concept
                if not concept_selector.has_concept_available():
                    continue

            concept = concept_selector.get_next_concept()

            examples, dataset = self.trainer.construct_dataset_for_concept_training(
                concept, use_concepts_as_negatives=use_concepts_as_negatives
            )

            # Create reader and writer locks for this new process
            paths = [example.image_features_path for example in examples]
            path_to_reader_lock = lock_generator.get_path_to_lock_mapping(paths, is_reader=True)
            path_to_writer_lock = lock_generator.get_path_to_lock_mapping(paths, is_reader=False)

            dataset.path_to_lock = path_to_reader_lock # Dataset only reads

            # Recache zero-shot attributes for sampled examples
            self.cacher.recache_zs_attr_features(
                concept,
                examples=examples,
                path_to_reader_lock=path_to_reader_lock,
                path_to_writer_lock=path_to_writer_lock
            )

            if self.use_concept_predictors_for_concept_components:
                for component in concept.component_concepts.values():
                    self.cacher.recache_zs_attr_features(
                        component,
                        examples=examples,
                        path_to_reader_lock=path_to_reader_lock,
                        path_to_writer_lock=path_to_writer_lock
                    )  # Necessary to predict component concepts

            else:  # Using fixed scores for concept-image pairs
                self.cacher.recache_component_concept_scores(
                    concept,
                    examples=examples,
                    path_to_reader_lock=path_to_reader_lock,
                    path_to_writer_lock=path_to_writer_lock
                )

            # Launch training
            next_device = available_devices.pop()
            logger.info(f'Launching training for concept "{concept.name}" on {next_device}')

            process = mp.Process(
                target=_launch_training,
                kwargs={
                    'trainer': trainer,
                    'concept_name': concept.name,
                    'examples': examples,
                    'dataset': dataset,
                    'return_queue': return_queue,
                    'n_epochs': n_epochs,
                    'device': next_device,
                    **train_concept_kwargs
                }
            )
            process.start()
            all_processes.append(process)

        # Ensure all processes are cleaned up
        for process in all_processes:
            process.join()

        manager.shutdown()

        logger.info(f"Time to train concepts: {time.time() - start_time:.2f}s")

        # Return all concepts to their original devices
        for concept_name, device in orig_devices.items():
            self.concept_kb[concept_name].predictor.to(device)

        return self.concept_kb