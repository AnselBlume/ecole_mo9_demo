import torch
import torch.nn as nn
from typing import Callable, Any
from dataclasses import dataclass
from model.features import ImageFeatures

@dataclass
class WeightedPredictorOutput:
    raw_scores: torch.Tensor
    weighted_scores: torch.Tensor
    final_score: torch.Tensor

class WeightedPredictor(nn.Module):
    def __init__(self, predictors: list, weights: torch.Tensor, name: str = ''):
        super().__init__()
        self.predictors = predictors
        self.weights = nn.Parameter(weights)
        self.name = name

    def forward(self, input: ImageFeatures, scores: torch.Tensor = None) -> torch.Tensor:
        if scores is None:
            scores = self.get_scores(input)

        weighted_scores = scores * self.weights

        return WeightedPredictorOutput(
            raw_scores=scores,
            weighted_scores=weighted_scores,
            final_score=weighted_scores.sum()
        )

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