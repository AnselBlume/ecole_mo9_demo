import yaml
from dataclasses import dataclass
from model.concept import ConceptKB, Concept
import logging
import json

logger = logging.getLogger(__file__)

@dataclass
class ConceptGraph:
    concepts: list[str]
    instance_graph: dict[str,str]
    component_graph: dict[str,str]

    def apply(self, concept_kb: ConceptKB):
        self.apply_concepts(concept_kb)
        self.apply_instance_graph(concept_kb)
        self.apply_component_graph(concept_kb)

    def apply_concepts(self, concept_kb: ConceptKB):
        graph_concepts = dict.fromkeys(self.concepts)

        for concept_name in graph_concepts: # Ensure all concepts in the graph are in the KB
            if concept_name not in concept_kb:
                logger.debug(f'Adding concept {concept_name} to KB')
                concept_kb.add_concept(Concept(concept_name))

        for concept in concept_kb: # Remove concepts not in the graph
            if concept.name not in graph_concepts:
                logger.debug(f'Removing concept {concept.name} from KB')
                concept_kb.remove_concept(concept)

    def apply_instance_graph(self, concept_kb: ConceptKB):
        for concept in concept_kb:
            concept.child_concepts.clear()
            concept.parent_concepts.clear()

        concept_set = set(self.concepts)
        for concept_name, children in self.instance_graph.items():
            if concept_name not in concept_set:
                logger.debug(f'Parent concept {concept_name} not in concept set but is in instance graph; skipping')
                continue

            concept = concept_kb[concept_name]

            for child_name in children:
                if child_name not in concept_set:
                    logger.debug(f'Child concept {child_name} not in concept set but is in instance graph; skipping')
                    continue

                child = concept_kb[child_name]
                concept.add_child_concept(child)

    def apply_component_graph(self, concept_kb: ConceptKB):
        for concept in concept_kb:
            concept.component_concepts.clear()
            concept.containing_concepts.clear()

        concept_set = set(self.concepts)
        for concept_name, components in self.component_graph.items():
            if concept_name not in concept_set:
                logger.debug(f'Containing concept {concept_name} not in concept set but is in component graph; skipping')
                continue

            concept = concept_kb[concept_name]

            for component_name in components:
                if component_name not in concept_set:
                    logger.debug(f'Component concept {component_name} not in concept set but is in component graph; skipping')
                    continue

                component = concept_kb[component_name]
                concept.add_component_concept(component)

class ConceptGraphParser:
    CONCEPTS_KEY = 'concepts'
    INSTANCE_GRAPH_KEY = 'instance_graph'
    COMPONENT_GRAPH_KEY = 'component_graph'

    def parse_graph(self, path: str = None, string: str = None, config: dict = None) -> ConceptGraph:
        assert bool(path) + bool(string) + bool(config) == 1, 'Exactly one of path or string must be provided'

        if path:
            with open(path, 'r') as f:
                try:
                    hierarchy_dict = yaml.safe_load(f)
                except:
                    hierarchy_dict = json.load(f)
        elif string:
            try:
                hierarchy_dict = yaml.safe_load(string)
            except:
                hierarchy_dict = json.load(string)
        else:
            hierarchy_dict = config

        concepts = hierarchy_dict.get(self.CONCEPTS_KEY, [])
        instance_graph = hierarchy_dict.get(self.INSTANCE_GRAPH_KEY, {})
        component_graph = hierarchy_dict.get(self.COMPONENT_GRAPH_KEY, {})

        return ConceptGraph(concepts=concepts, instance_graph=instance_graph, component_graph=component_graph)

'''
concepts:
    - airplane
    - commercial aircraft
    - cargo jet
    - passenger jet
    - biplane
    - fighter jet

    # Component concepts
    - propulsion component
    - wings
    - wing-mounted engine
    - bulky fuselage
    - row of windows
    - double wings
    - fixed landing gear
    - propeller
    - afterburner
    - openable nose

instance_graph:
    airplane: ['commercial aircraft', 'biplane', 'fighter jet']
    commercial aircraft: ['cargo jet', 'passenger jet']
    propulsion component: ['wing-mounted engine', 'afterburner', 'propeller']

component_graph:
    airplane: ['wings', 'propulsion component']
    commercial aircraft: ['wing-mounted engine']
    cargo jet: ['bulky fuselage', 'openable nose']
    pasenger jet: ['row of windows']
    biplane: ['double wings', 'fixed landing gear', 'propeller']
    fighter jet: ['afterburner']
'''