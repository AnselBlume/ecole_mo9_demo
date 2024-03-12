from __future__ import annotations
import pickle
from dataclasses import dataclass, field
from model.attribute import Attribute
from .concept_predictor import ConceptPredictor
from llm import LLMClient, retrieve_attributes
from tqdm import tqdm
import logging
from utils import ArticleDeterminer
from typing import Union, Any
from image_processing import LocalizeAndSegmentOutput
from .features import ImageFeatures
from PIL.Image import Image

logger = logging.getLogger(__name__)

@dataclass
class ConceptExample:
    image: Image = field(
        default=None,
        metadata={'description': 'Example\'s image'}
    )

    image_path: str = field(
        default=None,
        metadata={'description': 'Path to this example\'s image'}
    )

    image_segmentations: list[LocalizeAndSegmentOutput] = field(
        default=None,
        metadata={'description': 'Segmentations for the example'}
    )

    image_segmentations_path: list[str] = field(
        default=None,
        metadata={'description': 'Paths to segmentations for the example'}
    )

    are_segmentations_dirty: bool = False # Whether the segmentations need to be recomputed (e.g. due to a changed seg method)

    image_features: ImageFeatures = field(
        default=None,
        metadata={'description': 'Image features for the example'}
    )

    image_features_path: str = field(
        default=None,
        metadata={'description': 'Path to image features for this example'}
    )

    are_features_dirty: bool = False # Whether the features need to be recomputed (e.g. due to changed zs attributes)

    feature_extractor_id: str = field(
        default=None,
        metadata={'description': 'String uniquely identifying the feature extractor used to extract the features'}
    )

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

    examples: list[ConceptExample] = field(default_factory=list, metadata={'help': 'Stored example data'})

@dataclass
class ConceptKBConfig:
    encode_class_in_zs_attr: bool = False
    img_feature_dim: int = 768
    n_trained_attrs: int = None
    use_ln: bool = True
    use_probabilities: bool = False # Sigmoid scores instead of using raw scores for concept predictor inputs
    use_full_img: bool = True
    use_regions: bool = True

    def __post_init__(self):
        if not self.use_full_img and not self.use_regions:
            raise ValueError('At least one of use_full_img and use_regions must be True.')

        if self.use_ln and self.use_probabilities:
            raise ValueError('Cannot use both layer norm and probabilities.')

class ConceptKB:
    def __init__(self, concepts: list[Concept] = []):
        self._concepts = {concept.name : concept for concept in concepts}

    @property
    def leaf_concepts(self) -> list[Concept]:
        return [concept for concept in self.concepts if not concept.child_concepts]

    @property
    def root_concepts(self) -> list[Concept]:
        return [concept for concept in self.concepts if not concept.parent_concepts]

    @property
    def concepts(self) -> list[Concept]:
        return self.get_concepts()

    def children_of(self, concepts: list[Concept]) -> list[Concept]:
        children = {}

        # Construct children in ordered manner
        for concept in concepts:
            for child_name, child in concept.child_concepts.items():
                if child_name not in children:
                    children[child_name] = child

        return list(children.values())

    def rooted_subtree(self, concept: Concept) -> list[Concept]:
        '''
            Returns the rooted subtree of a concept.
        '''
        subtree = {}
        queue = [concept]

        while queue:
            curr = queue.pop()
            if curr.name not in subtree:
                subtree[curr.name] = curr
                queue.extend(curr.child_concepts.values())

        return list(subtree.values())

    def parameters(self):
        '''
            Returns all parameters of the concept predictors.
        '''
        params = []
        for concept in self.concepts:
            params.extend(list(concept.predictor.parameters()))

        return params

    def train(self):
        '''
            Sets all concept predictors to train mode.
        '''
        for concept in self.concepts:
            concept.predictor.train()

    def eval(self):
        '''
            Sets all concept predictors to eval mode.
        '''
        for concept in self.concepts:
            concept.predictor.eval()

    def to(self, device):
        '''
            Calls the to method on all concept predictors.
        '''
        for concept in self.concepts:
            concept.predictor.to(device)

    def save(self, path):
        '''
            Saves the ConceptKB to a pickle file.
        '''
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> ConceptKB:
        '''
            Returns a saved ConceptKB from a pickle file.
        '''
        with open(path, 'rb') as f:
            dt = pickle.load(f)

        return dt

    def initialize(
        self,
        cfg: ConceptKBConfig,
        llm_client: LLMClient = None,
        concept_to_zs_attrs: dict[str,Any] = None
    ):
        '''
            Initializes the ConceptKB with zero-shot attributes and concept predictors.
        '''
        self.cfg = cfg

        # Determine if we've already, or will, use zero-shot attributes from an LLM
        if not hasattr(self, 'used_zs_attrs_from_llm'):
            self.used_zs_attrs_from_llm = llm_client is not None

        # Get zero-shot attributes
        if llm_client is not None:
            self._init_zs_attrs(llm_client, cfg.encode_class_in_zs_attr, concept_to_zs_attrs=concept_to_zs_attrs)

        # Build predictors
        self._init_predictors()

    def _init_predictors(self):
        logger.info('Initializing concept predictors')

        for concept in tqdm(self.concepts):
            self.init_predictor(concept)

    def init_predictor(self, concept: Concept):
        concept.predictor = ConceptPredictor(
            img_feature_dim=self.cfg.img_feature_dim,
            region_feature_dim=self.cfg.img_feature_dim,
            n_trained_attrs=self.cfg.n_trained_attrs,
            n_zs_attrs=len(concept.zs_attributes),
            use_ln=self.cfg.use_ln,
            use_probabilities=self.cfg.use_probabilities,
            use_full_img=self.cfg.use_full_img,
            use_regions=self.cfg.use_regions
        )

    def _init_zs_attrs(
        self,
        llm_client: LLMClient,
        encode_class: bool,
        concept_to_zs_attrs: dict[str,Any] = None
    ):
        '''
            Initializes zero-shot attributes for ConceptKB's concepts.
        '''
        logger.info('Initializing zero-shot attributes from an LLM')

        determiner = ArticleDeterminer()
        for concept in tqdm(self.concepts):
            zs_attr_dict = concept_to_zs_attrs.get(concept.name, None) if concept_to_zs_attrs else None

            self.init_zs_attrs(
                concept,
                llm_client,
                encode_class,
                determiner=determiner,
                zs_attr_dict=zs_attr_dict
            )

    def init_zs_attrs(
        self,
        concept: Concept,
        llm_client: LLMClient,
        encode_class: bool,
        determiner: ArticleDeterminer = None,
        zs_attr_dict: dict[str,Any] = None
    ):
        '''
            Initializes zero-shot attributes for a specified Concept.
        '''
        if determiner is None and encode_class:
            determiner = ArticleDeterminer()

        zs_attr_dict = zs_attr_dict if zs_attr_dict else retrieve_attributes(concept.name, llm_client)

        for attr_type in ['required', 'likely']:
            for attr in zs_attr_dict[attr_type]:
                query = f'{attr} of {determiner.determine(attr)}{concept.name}' if encode_class else attr
                concept.zs_attributes.append(Attribute(attr, is_necessary=attr_type == 'required', query=query))

    def add_concept(self, concept: Union[str, Concept]):
        if isinstance(concept, str):
            concept = Concept(name=concept)

        self._concepts[concept.name] = concept

        return concept

    def remove_concept(self, concept: Union[str, Concept]):
        name = concept if isinstance(concept, str) else concept.name
        return self._concepts.pop(name)

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