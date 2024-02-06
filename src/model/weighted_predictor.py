import torch
import torch.nn as nn
from typing import Callable, Any
from attribute import Attribute
from concept import Concept

class WeightedPredictor(nn.Module):
    def __init__(self, predictors: list, weights: torch.Tensor, name: str = ''):
        self.predictors = predictors
        self.weights = nn.Parameter(weights)
        self.name = name

    def forward(self, input) -> torch.Tensor:
        scores = self.get_scores(input)
        return scores * self.weights

    def get_scores(self, input):
        raise NotImplementedError()

    def add_predictor(self, predictor, weight: float):
        self.predictors.append(predictor)
        self.weights = nn.Parameter(torch.cat([self.weights, torch.tensor(weight)]))

    def remove_predictor(self, filter_fn: Callable[[Any], bool] = None, index = None):
        def remove_at_index(index):
            self.predictors.pop(index)
            self.weights = nn.Parameter(torch.cat([self.weights[:index], self.weights[index+1:]]))

        if index is not None:
            remove_at_index(index)

        else:
            if filter_fn is None:
                raise ValueError('Either filter_fn or index must be provided.')

            for i, p in enumerate(self.predictors):
                if filter_fn(p):
                    remove_at_index(i)
                    return

class AttributeGroup(nn.Module):
    def __init__(self, attrs: list[Attribute], weights: torch.Tensor, name: str = ''):
        super().__init__(attrs, weights, name)

class ConceptGroup(nn.Module):
    def __init__(self, concepts: list[Concept], weights: torch.Tensor, name: str = ''):
        super().__init__(concepts, weights, name)