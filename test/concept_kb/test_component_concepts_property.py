import os
import sys
sys.path.append(os.path.realpath(__file__ + '/../../../src'))
from model.concept import Concept, ConceptKB

def test_component_concepts_property():
    '''
        Component relations:

        A --> A1, A2
        A2 --> A3
        B -> B1
        C
        D -> D1

        Instance Relations:
            A --> A3
            A3 --> A4
            A4 --> A5
            B --> B1, B2
            C -> C1
    '''
    A5 = Concept('A5')
    A4 = Concept('A4')
    A3 = Concept('A3')
    A2 = Concept('A2')
    A1 = Concept('A1')
    A = Concept('A')

    A.add_component_concept(A1)
    A.add_component_concept(A2)
    A2.add_component_concept(A3)

    A.add_child_concept(A3)
    A3.add_child_concept(A4)
    A4.add_child_concept(A5)

    B2 = Concept('B2')
    B1 = Concept('B1')
    B = Concept('B')

    B.add_component_concept(B1)

    B.add_child_concept(B1)
    B.add_child_concept(B2)

    C = Concept('C')
    C1 = Concept('C1')
    C.add_child_concept(C1)

    D = Concept('D')
    D1 = Concept('D1')
    D.add_component_concept(D1)

    concepts = [A5, A4, A3, A2, A1, A, B1, B2, B, C, C1, D, D1]
    concept_kb = ConceptKB(concepts=concepts)

    assert set(concept_kb.component_concepts) == {A1, A2, A3, A4, A5, B1, D1}