from __future__ import annotations
from dataclasses import dataclass, field
from model.attribute import Attribute
from .concept_predictor import ConceptPredictor
import logging
from image_processing import LocalizeAndSegmentOutput
from model.features import ImageFeatures
from PIL.Image import Image

logger = logging.getLogger(__name__)

@dataclass
class ConceptExample:
    concept_name: str = field(
        default=None,
        metadata={'description': 'Name of the concept that this example belongs to'}
    )

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

    is_negative: bool = field(
        default=False,
        metadata={'description': 'Whether this example is a negative example for its containing concept'}
    )

@dataclass
class Concept:
    name: str = ''

    # str --> Concept instead of list[Concept] to allow different namings of the same concept
    # Instance relations
    parent_concepts: dict[str,Concept] = field(
        default_factory=dict,
        metadata={'help': 'Dictionary of parent concepts. Keys are parent names (as described by this Concept), values are parent concepts.'}
    )
    child_concepts: dict[str,Concept] = field(
        default_factory=dict,
        metadata={'help': 'Dictionary of child concepts. Keys are child names (as described by this Concept), values are child concepts.'}
    )

    # Component relations
    containing_concepts: dict[str,Concept] = field(
        default_factory=dict,
        metadata={'help': 'Dictionary of containing concepts. Keys are containing concept names (as described by this Concept), values are containing concepts.'}
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

    def __hash__(self) -> int:
        return hash(self.name)

    def add_parent_concept(self, parent: Concept):
        self.parent_concepts[parent.name] = parent
        parent.child_concepts[self.name] = self

    def add_child_concept(self, child: Concept):
        self.child_concepts[child.name] = child
        child.parent_concepts[self.name] = self

    def add_containing_concept(self, containing: Concept):
        self.containing_concepts[containing.name] = containing
        containing.component_concepts[self.name] = self

    def add_component_concept(self, component: Concept):
        self.component_concepts[component.name] = component
        component.containing_concepts[self.name] = self