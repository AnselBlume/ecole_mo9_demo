from __future__ import annotations
import pickle
from dataclasses import dataclass, field
from model.attribute import Attribute
from .concept_predictor import ConceptPredictor
from llm import LLMClient, retrieve_attributes
from tqdm import tqdm
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

@dataclass
class ConceptKBConfig:
    encode_class_in_zs_attr: bool = False
    img_feature_dim: int = 768
    n_trained_attrs: int = None
    use_ln: bool = True
    use_full_img: bool = True

class ConceptKB:
    def __init__(self, concepts: list[Concept] = []):
        self._concepts = {concept.name : concept for concept in concepts}

    @property
    def concepts(self) -> list[Concept]:
        return self.get_concepts()

    def parameters(self):
        params = []
        for concept in self.concepts:
            params.extend(list(concept.predictor.parameters()))

        return params

    def train(self):
        for concept in self.concepts:
            concept.predictor.train()

    def eval(self):
        for concept in self.concepts:
            concept.predictor.eval()

    def to(self, device):
        for concept in self.concepts:
            concept.predictor.to(device)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> ConceptKB:
        with open(path, 'rb') as f:
            dt = pickle.load(f)

        assert isinstance(dt, ConceptKB)
        return dt

    def initialize(self, cfg: ConceptKBConfig, llm_client: LLMClient = None):
        self.cfg = cfg
        self.used_zs_attrs_from_llm = llm_client is not None

        # Get zero-shot attributes
        if llm_client is not None:
            self._init_zs_attrs(llm_client, cfg.encode_class_in_zs_attr)

        # Build predictors
        self._init_predictors()

    def _init_predictors(self):
        logger.info('Initializing concept predictors')

        for concept in tqdm(self.concepts):
            concept.predictor = ConceptPredictor(
                img_feature_dim=self.cfg.img_feature_dim,
                region_feature_dim=self.cfg.img_feature_dim,
                n_trained_attrs=self.cfg.n_trained_attrs,
                n_zs_attrs=len(concept.zs_attributes),
                use_ln=self.cfg.use_ln,
                use_full_img=self.cfg.use_full_img
            )

    def _init_zs_attrs(self, llm_client: LLMClient, encode_class: bool):
        logger.info('Initializing zero-shot attributes from an LLM')

        determiner = ArticleDeterminer()

        for concept in tqdm(self.concepts):
            zs_attr_dict = retrieve_attributes(concept.name, llm_client)

            for attr_type in ['required', 'likely']:
                for attr in zs_attr_dict[attr_type]:
                    query = f'{attr} of {determiner.determine(attr)}{concept.name}' if encode_class else attr
                    concept.zs_attributes.append(Attribute(attr, is_necessary=attr_type == 'required', query=query))

    def add_concept(self, concept: Concept):
        self._concepts[concept.name] = concept

    def remove_concept(self, name: str):
        self._concepts.pop(name)

    def get_concept(self, name: str) -> Concept:
        return self._concepts[name]

    def get_concepts(self) -> list[Concept]:
        return sorted(list(self._concepts.values()), key=lambda x: x.name)

    def __iter__(self):
        return iter(self.get_concepts())

    def __contains__(self, name: str):
        return name in self._concepts

    def __getitem__(self, name: str):
        return self.get_concept(name)

    def __len__(self):
        return len(self._concepts)