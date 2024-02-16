import os
from model.concept import Concept, ConceptKB

def label_from_path(path):
    return os.path.basename(path).split('_')[0].lower()

def kb_from_img_dir(img_dir: str) -> ConceptKB:
    '''
        Constructs a concept knowledge base from images in a directory.

        Arguments:
            img_dir (str): Directory containing images.

        Returns: ConceptKB
    '''
    kb = ConceptKB()

    for img in os.listdir(img_dir):
        label = label_from_path(img)

        if label not in kb:
            kb.add_concept(Concept(label))

        kb.get_concept(label).examples.append(img)

    return kb
