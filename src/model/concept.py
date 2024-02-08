from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from model.attribute import Attribute, AttributeGroup
from model.weighted_predictor import WeightedPredictor, WeightedPredictorOutput
import torch.nn as nn
import torch
from model.image_input import ImageInput

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

    attr_groups: dict[str, list[Attribute]] = field(
        default_factory=dict,
        metadata={'help': 'Dictionary of attribute groups. Keys are canonicalized group names, values are lists of attributes.'}
    )

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

        # Attribute detectors: necessary/descriptive, zero-shot/learned
        self.attr_groups = nn.ModuleDict({
            name : AttributeGroup(attrs, torch.ones(len(attrs)))
            for name, attrs in concept.attr_groups.items()
        })

        # TODO Calculation with this in forward can be optimized by using a single tensor
        self.attr_group_weights = nn.ModuleDict({
            name : nn.Parameter(torch.tensor(1. / len(concept.attr_groups)))
            for name in concept.attr_groups
        }) # Each attr group gets equal weight to start, with total weight == 1

        # Component concept detectors
        self.component_concepts = ConceptGroup(
            concept.component_concepts,
            torch.ones(len(concept.component_concepts))
        )

        self.component_concepts_weight = nn.Parameter(torch.tensor(1.))

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

@dataclass
class ConceptDB:
    concepts: dict[str, Concept] = field(
        default_factory=dict,
        metadata={'help': 'Dictionary of concepts. Keys are concept names, values are concepts.'}
    )

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