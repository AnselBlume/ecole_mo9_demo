from dataclasses import dataclass
import torch
from model.weighted_predictor import WeightedPredictor

@dataclass
class Attribute:
    name: str = ''
    is_necessary: bool = False
    query: str = ''

class ZeroShotAttribute(Attribute):
    pass

class LearnedAttribute(Attribute):
    pass

class AttributeGroup(WeightedPredictor):
    def __init__(self, attrs: list[Attribute], weights: torch.Tensor, name: str = ''):
        super().__init__(attrs, weights, name)
