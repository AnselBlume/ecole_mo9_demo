from dataclasses import dataclass

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

    zs_attrs: list[ZeroShotAttribute] = []
    learned_attrs: list[LearnedAttribute] = []

    zs_attr_weights: list[float] = []
    zs_attr_score_weight: float = .5

    learned_attr_weights: list[float] = []
    learned_attr_score_weight: float = .5

    def add_zs_attr(self, attr: ZeroShotAttribute, weight: float = 1.0):
        self.zs_attrs.append(attr)
        self.zs_attr_weights.append(weight)

    def add_learned_attr(self, attr: LearnedAttribute, weight: float = 1.0):
        self.learned_attrs.append(attr)
        self.learned_attr_weights.append(weight)

    def remove_zs_attr(self, name: str):
        for i, attr in enumerate(self.zs_attrs):
            if attr.name == name:
                self.zs_attrs.pop(i)
                self.zs_attr_weights.pop(i)
                break

    def remove_learned_attr(self, name: str):
        for i, attr in enumerate(self.learned_attrs):
            if attr.name == name:
                self.learned_attrs.pop(i)
                self.learned_attr_weights.pop(i)
                break

@dataclass
class ConceptSet:
    concepts: dict[str, Concept] = {}

    def add_concept(self, concept: Concept):
        self.concepts[concept.name] = concept

    def remove_concept(self, name: str):
        self.concepts.pop(name)

    def get_concept(self, name: str) -> Concept:
        return self.concepts[name]