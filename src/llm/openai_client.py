# %%
from openai import OpenAI
from .prompts import DEFAULT_SYS_PROMPT
import os
from typing import Union, Iterable

# Model pricing at https://openai.com/pricing
DEFAULT_MODEL= 'gpt-3.5-turbo-0125'

class LLMClient:
    def __init__(
            self,
            model: str = DEFAULT_MODEL,
            api_key: str = None
        ):
        self.model = model
        self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', api_key))

    def query(self, prompt: str, sys_prompt=DEFAULT_SYS_PROMPT, stream=False) -> Union[str, Iterable[str]]:
        '''
            Fetches the response from the LLM to the prompt using the OpenAI API, returning the response as
            a single string if stream is False, otherwise returns an iterable of strings.

            See https://cookbook.openai.com/examples/how_to_stream_completions for more info on streaming.
        '''
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            stream=stream
        )

        # Refer to https://cookbook.openai.com/examples/how_to_stream_completions for streaming
        if stream:
            return map(lambda c: c.choices[0].delta.content, response) # Extract text

        return response.choices[0].message.content
# %%
