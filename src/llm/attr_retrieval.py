# %%
from .openai_client import LLMClient
from .prompts import LIST_ATTRIBUTES_PROMPT
import re

import logging
logger = logging.getLogger(__name__)

# %%
def retrieve_attributes(concept_name: str, client: LLMClient) -> list[str]:
    '''
        Retrieves parts of a concept using LLM.
        Arguments:
            concept_name (str): Name of concept to retrieve parts for
            client (LLMClient): OpenAI LLM client
    '''
    response = client.query(LIST_ATTRIBUTES_PROMPT + concept_name)
    lines = response.split('\n')

    # Match a sequence of words and spaces, possibly preceded by a hyphen
    pattern = re.compile('\s*-?\s*((\w|\s)*\w)\s*')

    extracted_attrs = {'required': [], 'likely': []}
    is_likely = False

    for line in lines:
        line = line.lower().strip()

        if line == '' or line == 'none' or 'required' in line:
            continue

        if 'likely' in line:
            is_likely = True
            continue

        if (match := re.match(pattern, line)) is None:
            logger.warning(f'Could not extract attribute from response line "{line}"')
            continue

        extracted_attr = match.group(1)
        key = 'likely' if is_likely else 'required'
        extracted_attrs[key].append(extracted_attr)

    return extracted_attrs

# %%
