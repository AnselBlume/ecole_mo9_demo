# %%
import os # Change DesCo CUDA device here

# Prepend to path so starts searching at src first
import sys
sys.path = [os.path.join(os.path.dirname(__file__), '../src')] + sys.path

import json
from llm import LLMClient
from model.concept import ConceptKBConfig, ConceptKB, Concept
import logging, coloredlogs
from feature_extraction.trained_attrs import N_ATTRS_DINO
from kb_ops.build_kb import CONCEPT_TO_ATTRS_PATH

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

# %%
if __name__ == '__main__':
    #  Initialize concept KB
    concept_kb = ConceptKB()
    concept_kb.add_concept(Concept(name='husky'))

    # Load attributes from file
    with open(CONCEPT_TO_ATTRS_PATH) as f:
        concept_to_attrs = json.load(f)

    # %%
    concept_kb.initialize(ConceptKBConfig(
        n_trained_attrs=N_ATTRS_DINO,
    ), llm_client=LLMClient(), concept_to_zs_attrs=concept_to_attrs)

    # %% Print concept attributes
    print(concept_kb['husky'].zs_attributes)
# %%
