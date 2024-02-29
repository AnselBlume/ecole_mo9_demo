import os
from model.concept import Concept, ConceptKB, StoredExample

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

    for img_path in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_path)
        label = label_from_path(img_path)

        if label not in kb:
            kb.add_concept(Concept(label))

        kb.get_concept(label).examples.append(StoredExample(image_path=img_path))

    return kb
