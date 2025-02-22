from dataclasses import dataclass
import torch

@dataclass
class Attribute:
    name: str = ''
    is_necessary: bool = False
    query: str = ''

class ZeroShotAttribute(Attribute):
    pass

class LearnedAttribute(Attribute):
    pass
