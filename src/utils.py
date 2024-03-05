import torch
from typing import Union
import inflect

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
        return self.p.singular_noun(noun)