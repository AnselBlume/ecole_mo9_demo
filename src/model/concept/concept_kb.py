from __future__ import annotations
from dataclasses import dataclass
from torch.nn import Parameter
from model.attribute import Attribute
from itertools import chain
from .concept_predictor import ConceptPredictor
from collections import deque
import pickle
from tqdm import tqdm
import logging
from utils import ArticleDeterminer
from typing import Iterable, Union, Any
from .concept import Concept, ConceptExample
from llm import LLMClient, retrieve_attributes

logger = logging.getLogger(__name__)

@dataclass
class ConceptKBConfig:
    encode_class_in_zs_attr: bool = False
    include_descriptive_zs_attrs: bool = False
    img_feature_dim: int = 1024 # DINOv2 ViT-L/14 image feature dimension
    n_trained_attrs: int = None
    use_ln: bool = False # Layer norm the features before passing to ConceptPredictor
    use_probabilities: bool = False # Sigmoid scores instead of using raw scores for concept predictor inputs
    use_full_img: bool = True
    use_regions: bool = True
    use_region_features: bool = True # Use region features for class prediction, not just for region attr scores. use_regions must be true to have an effect

    def __post_init__(self):
        if not self.use_full_img and not self.use_regions:
            raise ValueError('At least one of use_full_img and use_regions must be True.')

        if self.use_ln and self.use_probabilities:
            raise ValueError('Cannot use both layer norm and probabilities.')

class ConceptKB:
    def __init__(self, concepts: list[Concept] = [], global_negatives: list[ConceptExample] = []):
        self._concepts = {concept.name : concept for concept in concepts}
        self.global_negatives = global_negatives

    @property
    def leaf_concepts(self) -> list[Concept]:
        '''
            Gets concepts which have no child concepts.
        '''
        return [concept for concept in self.concepts if not concept.child_concepts]

    @property
    def root_concepts(self) -> list[Concept]:
        '''
            Gets concepts which have no parent concepts.
        '''
        return [concept for concept in self.concepts if not concept.parent_concepts]

    @property
    def component_concepts(self) -> list[Concept]:
        '''
            Gets concepts which are components of another Concept (component relation), or which are
            descendants of a component concept via child relations.
        '''
        # Get all component concepts
        components = {} # Maintain order with dictionary
        for concept in self.concepts:
            components.update(dict.fromkeys(concept.component_concepts.values()))

        # Get all descendants of a component concept
        descendants_of_components = {}
        for component_concept in components:
            descendants = self.rooted_subtree(component_concept)
            descendants_of_components.update(dict.fromkeys(descendants))

        return list(descendants_of_components.keys())

    @property
    def concepts(self) -> list[Concept]:
        return self.get_concepts()

    def markov_blanket(self, concept: Concept, include_component_concept_roots: bool = False) -> list[Concept]:
        '''
            Returns the set of concepts which influence the prediction of this concept when traversing the graph from
            roots to leaves. This is not a true Markov blanket, but is similar in function.

            Specifically, the list includes the concept itself, its ancestors, its ancestors' siblings, and non-component
            root nodes (as its highest-level ancestor may not have siblings by definition of having a parent).

            If include_component_concepts is True, the list also includes the root nodes which are component concepts.
        '''
        # Add ancestors and the concept itself
        blanket = dict.fromkeys(self.rooted_subtree(concept, reverse_edges=True))

        # Add each ancestor's siblings
        blanket.update(dict.fromkeys(self.children_of([c for c in blanket if c.name != concept.name])))

        # Add root nodes
        if include_component_concept_roots: # All root nodes
            blanket.update(dict.fromkeys(self.root_concepts))
        else: # Non-component root nodes
            component_concepts = set(self.component_concepts)
            blanket.update({c : None for c in self.root_concepts if c not in component_concepts})

        return list(blanket)

    def children_of(self, concepts: list[Concept]) -> list[Concept]:
        children = {}

        # Construct children in ordered manner
        for concept in concepts:
            for child_name, child in concept.child_concepts.items():
                if child_name not in children:
                    children[child_name] = child

        return list(children.values())

    def rooted_subtree(self, concept: Concept, reverse_edges: bool = False) -> list[Concept]:
        '''
            Returns the rooted subtree of a concept. This is the concept itself and all its descendants.

            If reversed_edges is True, the rooted subtree is the concept itself and all its ancestors.
        '''
        subtree = {}
        queue = deque([concept])

        while queue:
            curr = queue.popleft()

            if curr.name not in subtree:
                subtree[curr.name] = curr

                if reverse_edges:
                    queue.extend(curr.parent_concepts.values())
                else:
                    queue.extend(curr.child_concepts.values())

        return list(subtree.values())

    def parameters(self) -> Iterable[Parameter]:
        '''
            Returns all parameters of the concept predictors.
        '''
        return chain.from_iterable(concept.predictor.parameters() for concept in self.concepts)

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

    def device(self) -> str:
        '''
            Returns the device of the first concept predictor.
        '''
        return next(iter(self.concepts)).predictor.device()
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
            n_component_concepts=len(concept.component_concepts),
            use_ln=self.cfg.use_ln,
            use_probabilities=self.cfg.use_probabilities,
            use_full_img=self.cfg.use_full_img,
            use_regions=self.cfg.use_regions,
            use_region_features=self.cfg.use_region_features
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

        attr_types = ['required'] + (['likely'] if self.cfg.include_descriptive_zs_attrs else [])
        for attr_type in attr_types:
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

    def get_concepts(self, in_component_order: bool = False) -> list[Concept]:
        if in_component_order:
            return self.in_component_order()

        return sorted(list(self._concepts.values()), key=lambda x: x.name)

    def in_component_order(self, concepts: list[Concept] = None) -> list[Concept]:
        '''
            Returns the concepts in the order of their component dependencies.
            I.e. if A has B as a component, B will come before A in the list.

            Note that if concepts are specified, this function only orders them in terms of their topological order
            based on component dependencies; it does not compute the complete set of concepts in the component subgraph.
        '''
        if concepts is not None:
            all_ordered_concepts = self.in_component_order()
            concept_names = {concept.name for concept in concepts}
            return [concept for concept in all_ordered_concepts if concept.name in concept_names]

        # Extract component concept dependencies for which the component concepts are in the actual ConceptKB
        # Build reversed adjacency list: if A has B as a component, B --> A
        adj_list = {concept.name : [] for concept in self.concepts}
        for concept in self.concepts:
            for component_name in concept.component_concepts:
                adj_list[component_name].append(concept.name)

        # Topological sort
        visited_set = set()
        visited_order = []
        def dfs(concept_name: str):
            if concept_name in visited_set:
                return

            visited_set.add(concept_name)

            for child_name in adj_list[concept_name]:
                dfs(child_name)

            visited_order.append(concept_name)

        for concept in self.concepts:
            dfs(concept.name)

        topological_names = list(reversed(visited_order))
        topological_concepts = [self.get_concept(concept_name) for concept_name in topological_names]

        return topological_concepts

    def get_component_concept_subgraph(self, concepts: list[Concept]) -> list[Concept]:
        '''
            Returns the smallest set of concepts containing the provided concepts and all concepts reachable
            from them via component relations.
        '''
        visited = {}
        def dfs(concept: Concept):
            if concept.name in visited:
                return

            visited[concept.name] = concept

            for child in concept.component_concepts.values():
                dfs(child)

        for concept in concepts:
            dfs(concept)

        return list(visited.values())

    def __iter__(self):
        return iter(self.get_concepts())

    def __contains__(self, name: str):
        return name in self._concepts

    def __getitem__(self, name: str):
        return self.get_concept(name)

    def __len__(self):
        return len(self._concepts)