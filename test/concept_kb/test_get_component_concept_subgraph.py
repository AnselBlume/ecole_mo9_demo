import os
import sys
sys.path.append(os.path.realpath(__file__ + '/../../../src'))
from model.concept import Concept, ConceptKB

'''
    Component relations:

    A --> A1, A2
    A2 --> A3
    B --> B1, B2
    C
'''
A3 = Concept('A3')
A2 = Concept('A2', component_concepts={'A3': A3})
A1 = Concept('A1')
A = Concept('A', component_concepts={'A1': A1, 'A2': A2})

B1 = Concept('B1')
B2 = Concept('B2')
B = Concept('B', component_concepts={'B1': B1, 'B2': B2})

C = Concept('C')

concepts = [A3, A2, A1, A, B1, B2, B, C]
concept_kb = ConceptKB(concepts=concepts)

def test_root():
    concepts = concept_kb.get_component_concept_subgraph(concepts=[concept_kb['A']])

    assert {concept.name for concept in concepts} == {'A', 'A1', 'A2', 'A3'}

def test_parent():
    concepts = concept_kb.get_component_concept_subgraph(concepts=[concept_kb['B']])

    assert {concept.name for concept in concepts} == {'B', 'B1', 'B2'}

def test_leaf():
    concepts = concept_kb.get_component_concept_subgraph(concepts=[concept_kb['A1']])

    assert {concept.name for concept in concepts} == {'A1'}