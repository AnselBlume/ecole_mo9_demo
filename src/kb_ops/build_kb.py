import os
from model.concept import Concept, ConceptKB, ConceptExample
from typing import Callable

def label_from_path(path):
    return os.path.basename(path).split('_')[0].lower()

def label_from_directory(path):
    return os.path.basename(os.path.dirname(path)).lower()

def kb_from_img_dir(
    img_dir: str,
    label_from_path_fn: Callable[[str],str] = label_from_path,
    exts: list[str] = ['.jpg', '.png']
) -> ConceptKB:
    '''
        Constructs a concept knowledge base from images in a directory.

        Arguments:
            img_dir (str): Directory containing images.

        Returns: ConceptKB
    '''
    kb = ConceptKB()
    exts = set(exts)

    for dirpath, dirnames, filenames in os.walk(img_dir):
        for filename in filenames:
            path = os.path.join(dirpath, filename)

            if os.path.splitext(path)[1] in exts:
                label = label_from_path_fn(path)

                if label not in kb:
                    kb.add_concept(Concept(label))

                kb.get_concept(label).examples.append(ConceptExample(image_path=path))

    return kb
