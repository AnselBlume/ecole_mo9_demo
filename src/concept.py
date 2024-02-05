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
    necsry_zs_attrs: list[ZeroShotAttribute] = []
    necsry_learned_attrs: list[LearnedAttribute] = []

    descr_zs_attrs: list[ZeroShotAttribute] = []
    descr_learned_attrs: list[LearnedAttribute] = []

    # Individual weights for necessary attributes
    necsry_zs_attr_weights: list[float] = [] # Zero shot, necessary attribute weights
    necsry_learned_attr_weights: list[float] = [] # Learned, necessary attribute weights

    # Aggregate weights for necessary attributes
    necsry_zs_attr_score_weight: float = .5
    necsry_learned_attr_score_weight: float = .5

    # Individual weights for descriptive attributes
    descr_zs_attr_weights: list[float] = []
    descr_learned_attr_weights: list[float] = []

    # Aggregate weights for descriptive attributes
    descr_zs_attr_score_weight: float = .5
    descr_learned_attr_score_weight: float = .5

    def add_zs_attr(self, attr: ZeroShotAttribute, weight: float):
        zs_attrs = self.necsry_zs_attrs if attr.is_necessary else self.descr_zs_attrs
        zs_attr_weights = self.necsry_zs_attr_weights if attr.is_necessary else self.descr_zs_attr_weights

        # Add in sorted order based on name
        idx = bisect.bisect_left([attr.name for attr in zs_attrs], attr.name)

        zs_attrs.insert(idx, attr)
        zs_attr_weights.insert(idx, weight)

    def remove_zs_attr(self, name: str):
        if name in [attr.name for attr in self.necsry_zs_attrs]: # Necessary attribute
            zs_attrs = self.necsry_zs_attrs
            zs_attr_weights = self.necsry_zs_attr_weights

        else: # Descriptive attribute
            zs_attrs = self.descr_zs_attrs
            zs_attr_weights = self.descr_zs_attr_weights

        idx = [attr.name for attr in zs_attrs].index(name)

        zs_attrs.pop(idx)
        zs_attr_weights.pop(idx)

    def add_learned_attr(self, attr: LearnedAttribute, weight: float):
        learned_attrs = self.necsry_learned_attrs if attr.is_necessary else self.descr_learned_attrs
        learned_attr_weights = self.necsry_learned_attr_weights if attr.is_necessary else self.descr_learned_attr_weights

        # Add in sorted order based on name
        idx = bisect.bisect_left([attr.name for attr in learned_attrs], attr.name)

        learned_attrs.insert(idx, attr)
        learned_attr_weights.insert(idx, weight)

    def remove_learned_attr(self, name: str):
        if name in [attr.name for attr in self.necsry_learned_attrs]:
            learned_attrs = self.necsry_learned_attrs
            learned_attr_weights = self.necsry_learned_attr_weights

        else:
            learned_attrs = self.descr_learned_attrs
            learned_attr_weights = self.descr_learned_attr_weights

        idx = [attr.name for attr in learned_attrs].index(name)

        learned_attrs.pop(idx)
        learned_attr_weights.pop(idx)

    def get_zs_attrs(self) -> list[ZeroShotAttribute]:
        return sorted(self.necsry_zs_attrs + self.descr_zs_attrs, key=lambda x: x.name)

    def get_learned_attrs(self) -> list[LearnedAttribute]:
        return sorted(self.necsry_learned_attrs + self.descr_learned_attrs, key=lambda x: x.name)

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