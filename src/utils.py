import torch
import inflect
import PIL.Image as Image
from PIL import ImageOps
from typing import Union

def open_image(img_path: str):
    '''
        Safely opens an image from a path and closes the file afterwards, rotating the image based on EXIF data.
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        img.load()  # Explicitly load image data before closing the file

    img = img.convert('RGB')

    # Rotate image based on EXIF data; prevents PIL from returning incorrect image orientation
    img = ImageOps.exif_transpose(img) # Rotate image based on EXIF data

    return img

def replace_extension(file_name: str):
    if '/' in file_name:
        return file_name.split('/')[-1].strip()
    else:
        return file_name

def to_device(d: dict, device: Union[str, torch.device]):
    '''
        Recursively moves all tensors in d to the device.
    '''
    for k, v in d.items():
        if isinstance(v, dict):
            to_device(v, device)
        elif isinstance(v, torch.Tensor):
            d[k] = v.to(device)

    return d

class ArticleDeterminer:
    def __init__(self):
        self.p = inflect.engine()

    def determine(self, word: str, space_if_nonempty: bool = True):
        '''
            Returns 'a/an' if the word is singular, else returns ''.

            If space_if_nonempty is True, appends a space to the article if it is nonempty, allowing for
            constructions like:
             ```python
             f'{article_determiner.determine(word)}{word}'
             ```
        '''
        # Determine the article for the concept
        is_singular_noun = not bool(self.p.singular_noun(word)) # Returns false if singular; singular version if plural
        article = self.p.a(word) if is_singular_noun else ''

        if article: # Split off the 'a' or 'an' from the word
            article = article.split(' ')[0]

        if space_if_nonempty and article:
            article += ' '

        return article

    def to_singular(self, noun: str):
        result = self.p.singular_noun(noun) # Returns False if already singular

        return result if result else noun