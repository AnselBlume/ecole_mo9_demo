from dataclasses import dataclass
import bisect

@dataclass
class Attribute:
    name: str = ''
    is_necessary: bool = False
    query: str = ''

class ZeroShotAttribute(Attribute):
    pass

class LearnedAttribute(Attribute):
    pass

@dataclass
class Concept:
    name: str = ''

    # TODO Make these all torch.Parameters

    # All attributes
    zs_attrs: list[ZeroShotAttribute] = []
    learned_attrs: list[LearnedAttribute] = []

    # Individual weights for necessary attributes
    nec_zs_attr_weights: list[float] = [] # Zero shot, necessary attribute weights
    nec_learned_attr_weights: list[float] = [] # Learned, necessary attribute weights

    # Aggregate weights for necessary attributes
    nec_zs_attr_score_weight: float = .5
    nec_learned_attr_score_weight: float = .5

    # Individual weights for descriptive attributes
    descr_zs_attr_weights: list[float] = []
    descr_learned_attr_weights: list[float] = []

    # Aggregate weights for descriptive attributes
    descr_zs_attr_score_weight: float = .5
    descr_learned_attr_score_weight: float = .5

    def add_zs_attr(self, zs_attr: ZeroShotAttribute, weight: float):
        # Add in sorted order based on name
        idx = bisect.bisect_left([attr.name for attr in self.zs_attrs], zs_attr.name)

        self.zs_attrs.insert(idx, zs_attr)
        self.zs_attr_weights.insert(idx, weight)

    def remove_zs_attr(self, name: str):
        idx = bisect.bisect_left([attr.name for attr in self.zs_attrs], name)

        self.zs_attrs.pop(idx)
        self.zs_attr_weights.pop(idx)

    def add_learned_attr(self, learned_attr: LearnedAttribute, weight: float):
        # Add in sorted order based on name
        idx = bisect.bisect_left([attr.name for attr in self.learned_attrs], learned_attr.name)

        self.learned_attrs.insert(idx, learned_attr)
        self.learned_attr_weights.insert(idx, weight)

    def remove_learned_attr(self, name: str):
        idx = bisect.bisect_left([attr.name for attr in self.learned_attrs], name)

        self.learned_attrs.pop(idx)
        self.learned_attr_weights.pop(idx)

@dataclass
class ConceptSet:
    concepts: dict[str, Concept] = {}

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