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

    Expected should be
        A3 before A2
        A1, A2 before A
        B1, B2 before B
        C anywhere
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

def test_in_component_order():
    sorted_concepts = concept_kb.in_component_order()
    sorted_names = [concept.name for concept in sorted_concepts]

    # Extract sorted indices
    a3_ind = sorted_names.index('A3')
    a2_ind = sorted_names.index('A2')
    a1_ind = sorted_names.index('A1')
    a_ind = sorted_names.index('A')

    b1_ind = sorted_names.index('B1')
    b2_ind = sorted_names.index('B2')
    b_ind = sorted_names.index('B')

    c_ind = sorted_names.index('C') # Raises value error if not present

    # Validate
    assert a3_ind < a2_ind
    assert a1_ind < a_ind
    assert a2_ind < a_ind

    assert b1_ind < b_ind
    assert b2_ind < b_ind

def test_in_component_order_specified_concepts():
    concepts_to_extract = [concept_kb['A1'], concept_kb['A2'], concept_kb['A3'], concept_kb['A']]
    sorted_concepts = concept_kb.in_component_order(concepts=concepts_to_extract)
    sorted_names = [concept.name for concept in sorted_concepts]

    # Extract sorted indices
    a3_ind = sorted_names.index('A3')
    a2_ind = sorted_names.index('A2')
    a1_ind = sorted_names.index('A1')
    a_ind = sorted_names.index('A')

    # Validate
    assert a3_ind < a2_ind
    assert a1_ind < a_ind
    assert a2_ind < a_ind
    assert len(sorted_names) == 4 # onnly these four returned