from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from model.attribute import Attribute
from model.weighted_predictor import WeightedPredictor, WeightedPredictorOutput
import torch.nn as nn
import torch
from llm import LLMClient, retrieve_attributes
from model.image_input import ImageInput
import logging
from utils import ArticleDeterminer

logger = logging.getLogger(__name__)

@dataclass
class Concept:
    name: str = ''

    # str --> Concept instead of list[Concept] to allow different namings of the same concept
    parent_concepts: dict[str,Concept] = field(
        default_factory=dict,
        metadata={'help': 'Dictionary of parent concepts. Keys are parent names (as described by this Concept), values are parent concepts.'}
    )
    child_concepts: dict[str,Concept] = field(
        default_factory=dict,
        metadata={'help': 'Dictionary of child concepts. Keys are child names (as described by this Concept), values are child concepts.'}
    )
    component_concepts: dict[str,Concept] = field(
        default_factory=dict,
        metadata={'help': 'Dictionary of component concepts. Keys are component names (as described by this Concept), values are component concepts.'}
    )

    zs_attributes: list[Attribute] = field(
        default_factory=list,
        metadata={'help': 'List of zero-shot attributes.'}
    )

    predictor: ConceptPredictor = field(
        default=None,
        metadata={'help': 'Predictor for this concept.'}
    )

    examples: list = field(default_factory=list, metadata={'help': 'Paths to example images.'})

class ConceptGroup(WeightedPredictor):
    def __init__(self, concepts: list[Concept], weights: torch.Tensor, name: str = ''):
        super().__init__(concepts, weights, name)

@dataclass
class ConceptPredictorOutput:
    attr_group_results: dict[str, WeightedPredictorOutput]
    component_concept_results: WeightedPredictorOutput
    final_score: torch.Tensor

class ConceptPredictor(nn.Module):
    # TODO incorporate unnamed visual features
    def __init__(self, concept: Concept):
        self.concept = concept
        self.feature_groups = nn.ModuleDict()

        # Attribute detectors: necessary/descriptive, zero-shot/learned

        # TODO detect component concept detectors

    def forward(self, input: ImageInput):
        # Get attribute scores
        attr_group_results = {
            name : group(input)
            for name, group in self.attr_groups.items()
        }

        weighted_attr_score = sum(
            attr_group_results[name].final_score * self.attr_group_weights[name]
            for name in attr_group_results
        )

        # Get component concept scores
        component_concept_results = self.component_concepts(input)
        weighted_concept_score = component_concept_results.final_score * self.component_concepts_weight

        # Final score
        final_score = weighted_attr_score + weighted_concept_score

        return ConceptPredictorOutput(
            attr_group_results=attr_group_results,
            component_concept_results=component_concept_results,
            final_score=final_score
        )

class ConceptKB:
    def __init__(self, concepts: list[Concept] = []):
        self.concepts = {concept.name : concept for concept in concepts}

    def initialize(
        self,
        llm_client: LLMClient = None,
        encode_class_in_zs_attr: bool = False
    ):
        # Get zero-shot attributes
        if llm_client:
            self._init_zs_attrs(llm_client, encode_class_in_zs_attr)

        # Build predictors

    def _init_zs_attrs(self, llm_client: LLMClient, encode_class: bool):
        determiner = ArticleDeterminer()

        for concept in self.concepts.values():
            zs_attr_dict = retrieve_attributes(concept.name, llm_client)

            for attr_type in ['required', 'likely']:
                for attr in zs_attr_dict[attr_type]:
                    query = f'{attr} of {determiner.determine(attr)}{concept.name}' if encode_class else attr
                    concept.zs_attributes.append(Attribute(attr, necessary=attr_type == 'required', query=query))

    def add_concept(self, concept: Concept):
        self.concepts[concept.name] = concept

    def remove_concept(self, name: str):
        self.concepts.pop(name)

    def get_concept(self, name: str) -> Concept:
        return self.concepts[name]

    def get_concepts(self) -> list[Concept]:
        return sorted(list(self.concepts.values()), key=lambda x: x.name)

    def __iter__(self):
        return iter(self.get_concepts())

    def __contains__(self, name: str):
        return name in self.concepts

    def __getitem__(self, name: str):
        return self.get_concept(name)