import json
from model.concept import ConceptKB
from kb_ops import ConceptKBTrainer, ConceptKBPredictor
from kb_ops.retrieve import CLIPConceptRetriever
from llm import LLMClient
from kb_ops.feature_pipeline import ConceptKBFeaturePipeline
from kb_ops.caching import ConceptKBFeatureCacher
from model.concept import Concept, ConceptExample
from utils import ArticleDeterminer
from visualization.heatmap import HeatmapVisualizer
import logging
from kb_ops.build_kb import CONCEPT_TO_ATTRS_PATH
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ControllerConfig:
    concept_to_zs_attrs_json_path: str = CONCEPT_TO_ATTRS_PATH
    cache_predictions: bool = False

class BaseController:
    def __init__(
        self,
        concept_kb: ConceptKB,
        feature_pipeline: ConceptKBFeaturePipeline,
        trainer: ConceptKBTrainer = None,
        predictor: ConceptKBPredictor = None,
        retriever: CLIPConceptRetriever = None,
        cacher: ConceptKBFeatureCacher = None,
        llm_client: LLMClient = None,
        heatmap_visualizer: HeatmapVisualizer = None,
        config: ControllerConfig = ControllerConfig()
    ):

        self.concept_kb = concept_kb
        self.feature_pipeline = feature_pipeline
        self.trainer = trainer if trainer else ConceptKBTrainer(concept_kb, self.feature_pipeline)
        self.predictor = predictor if predictor else ConceptKBPredictor(concept_kb, self.feature_pipeline)
        self.retriever = retriever if retriever else CLIPConceptRetriever(
            concept_kb.concepts,
            self.feature_pipeline.feature_extractor.clip,
            self.feature_pipeline.feature_extractor.processor
        )
        self.cacher = cacher if cacher else ConceptKBFeatureCacher(concept_kb, self.feature_pipeline, cache_dir='feature_cache')
        self.llm_client = llm_client if llm_client else LLMClient()
        self.heatmap_visualizer = heatmap_visualizer if heatmap_visualizer else HeatmapVisualizer(
            concept_kb,
            feature_pipeline.feature_extractor.dino_feature_extractor
        )
        self.config = config

        self.cached_predictions = []
        self.cached_images = []

        # Load external knowledgebase of concepts to zero-shot attributes
        if config.concept_to_zs_attrs_json_path:
            with open(config.concept_to_zs_attrs_json_path) as f:
                self.concept_to_zs_attrs = json.load(f)

        else:
            self.concept_to_zs_attrs = {}

    @property
    def use_concept_predictors_for_concept_components(self):
        return not self.feature_pipeline.config.compute_component_concept_scores

    def clear_cache(self):
        self.cached_predictions = []
        self.cached_images = []

    def retrieve_concept(self, concept_name: str, max_retrieval_distance: float = .5):
        concept_name = concept_name.strip()

        if concept_name in self.concept_kb:
            concept = self.concept_kb[concept_name]

        elif concept_name.lower() in self.concept_kb:
            concept = self.concept_kb[concept_name.lower()]

        else:
            retrieved_concept = self.retriever.retrieve(concept_name, 1)[0]
            logger.info(f'Retrieved concept "{retrieved_concept.concept.name}" with distance: {retrieved_concept.distance}')
            if retrieved_concept.distance > max_retrieval_distance:
                raise RuntimeError(f'No concept found for "{concept_name}".')

            concept = retrieved_concept.concept

        logger.info(f'Retrieved concept with name: "{concept.name}" for input "{concept_name}"')

        return concept

    ################################
    # Concept Addition and Removal #
    ################################
    def add_concept(
        self,
        concept_name: str = None,
        concept: Concept = None,
        parent_concept_names: list[str] = [],
        child_concept_names: list[str] = [],
        containing_concept_names: list[str] = [],
        component_concept_names: list[str] = [],
        use_singular_name: bool = False
    ):
        if not (bool(concept_name is None) ^ bool(concept is None)):
            raise ValueError('Exactly one of concept_name or concept must be provided.')

        if concept_name is not None: # Normalize the name
            concept_name = concept_name.lower()

            if use_singular_name:
                determiner = ArticleDeterminer()
                concept_name = determiner.to_singular(concept_name)

            concept = Concept(concept_name)

        if concept.name in self.concept_kb:
            raise ValueError(f'Concept with name "{concept.name}" already exists in the ConceptKB.')

        # Add relations if concept_name was provided instead of Concept object
        if concept_name is not None:
            # These setters set the parent and child / component and containing concepts simultaneously
            for parent_name in parent_concept_names:
                parent_concept = self.retrieve_concept(parent_name)
                concept.add_parent_concept(parent_concept)

            for child_name in child_concept_names:
                child_concept = self.retrieve_concept(child_name)
                concept.add_child_concept(child_concept)

            if containing_concept_names:
                logger.warning(f'Containing concepts "{containing_concept_names}" must be retrained after adding component "{concept.name}".')

            for containing_name in containing_concept_names:
                containing_concept = self.retrieve_concept(containing_name)
                concept.add_containing_concept(containing_concept)

            for component_name in component_concept_names:
                component_concept = self.retrieve_concept(component_name)
                concept.add_component_concept(component_concept)

        else:
            if parent_concept_names or child_concept_names or containing_concept_names or component_concept_names:
                raise ValueError('Relations can only be added if concept_name is provided instead of a Concept object.')

        # Get zero shot attributes (query LLM)
        self.concept_kb.init_zs_attrs(
            concept,
            self.llm_client,
            encode_class=self.concept_kb.cfg.encode_class_in_zs_attr,
            zs_attr_dict=self.concept_to_zs_attrs.get(concept.name, None)
        )

        self.concept_kb.init_predictor(concept)

        if len(self.concept_kb) and next(iter(self.concept_kb)).predictor is not None:
            concept.predictor.to(next(self.concept_kb.parameters()).device) # Assumes all concepts are on the same device
        else: # Assume there aren't any other initialized concept predictors
            concept.predictor.cuda()

        self.concept_kb.add_concept(concept)
        self.retriever.add_concept(concept)
        self.trainer.recompute_labels()
        self.predictor.recompute_labels()

        return concept

    def clear_concepts(self):
        for concept in self.concept_kb:
            self.concept_kb.remove_concept(concept.name)
            self.retriever.remove_concept(concept.name)

        self.trainer.recompute_labels()
        self.predictor.recompute_labels()

    def remove_concept(self, concept_name: str):
        try:
            concept = self.retrieve_concept(concept_name, max_retrieval_distance=0.)
        except RuntimeError as e:
            logger.info(f'No exact match for concept with name "{concept_name}". Not removing concept to be safe.')
            raise(e)

        self.concept_kb.remove_concept(concept.name)
        self.retriever.remove_concept(concept.name)
        self.trainer.recompute_labels()
        self.predictor.recompute_labels()

    ########################
    # Concept Modification #
    ########################
    def add_examples(self, examples: list[ConceptExample], concept_name: str = None, concept: Concept = None):
        if not (bool(concept_name is None) ^ bool(concept is None)):
            raise ValueError('Exactly one of concept_name or concept must be provided.')

        if concept is None:
            concept = self.retrieve_concept(concept_name)

        image_paths = {ex.image_path for ex in concept.examples}

        for example in examples:
            if not example.concept_name: # Ensure concept name is set
                example.concept_name = concept.name

            if example.image_path not in image_paths: # Ensure no duplicate examples for this concept
                concept.examples.append(example)

        return concept

    def add_hyponym(self, child_name: str, parent_name: str, child_max_retrieval_distance: float = 0.2):
        parent = self.retrieve_concept(parent_name)

        try:
            child = self.retrieve_concept(child_name, max_retrieval_distance=child_max_retrieval_distance)
        except RuntimeError:
            child = self.add_concept(child_name, parent_concept_names=[parent_name])

        child.add_parent_concept(parent) # This sets the parent's child pointer as well

    def add_component_concept(self, component_concept_name: str, concept_name: str, component_max_retrieval_distance: float = 0.2):
        concept = self.retrieve_concept(concept_name)

        try:
            component = self.retrieve_concept(component_concept_name, max_retrieval_distance=component_max_retrieval_distance)
        except RuntimeError:
            component = self.add_concept(component_concept_name)

        concept.add_component_concept(component)
        concept.predictor.set_num_component_concepts(len(concept.component_concepts))

    def add_concept_negatives(self, concept_name: str, negatives: list[ConceptExample]):
        assert all(negative.is_negative for negative in negatives), 'All ConceptExamples must have is_negative=True.'

        concept = self.retrieve_concept(concept_name)
        concept.examples.extend(negatives)