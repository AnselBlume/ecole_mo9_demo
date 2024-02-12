# %%
from .openai_client import LLMClient
from .prompts import LIST_PARTS_PROMPT
import re

import logging
logger = logging.getLogger(__name__)

# %%
def retrieve_parts(concept_name: str, client: LLMClient) -> list[str]:
    '''
        Retrieves parts of a concept using LLM.
        Arguments:
            concept_name (str): Name of concept to retrieve parts for
            client (LLMClient): OpenAI LLM client
    '''
    response = client.query(LIST_PARTS_PROMPT + concept_name)
    part_lines = response.split('\n')

    # Match a sequence of words and spaces, possibly preceded by a hyphen
    pattern = re.compile('\s*-?\s*((\w|\s)*\w)\s*')

    extracted_parts = []
    for part_line in part_lines:
        if (match := re.match(pattern, part_line)) is None:
            logger.warning(f'Could not extract part name from part response line "{part_line}"')
            continue

        extracted_part = match.group(1)
        extracted_parts.append(extracted_part)

        if extracted_part == concept_name:
            logger.info('LLM returned concept name part, indicating no component parts')
            break

    return extracted_parts

# %%
