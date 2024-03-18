# %%
import json
import os

DEFAULT_CKPT_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../attribute_training/classifiers_official.pth'
)

# Mapping from index to attribute name
with open(os.path.join(os.path.dirname(__file__), 'attribute_index.json')) as f:
    attr_to_index = json.load(f)

INDEX_TO_ATTR = {v: k for k, v in attr_to_index.items()}

# Load mapping from 'color', 'shape', and 'material' to their best subset of attributes indices
subset_index_path = os.path.join(
    os.path.dirname(__file__),
    '../../attribute_training/subset_color_shape_material.json'
)

with open(subset_index_path, 'r') as f:
    COLOR_SHAPE_MATERIAL_SUBSET = json.load(f)
# DINO mapping 
dino_index_path = os.path.join(
    os.path.dirname(__file__),
    'dino_class_id_to_index.json'
)
with open(dino_index_path, 'r') as f:
    DINO_INDEX_TO_ATTR = json.load(f)
DINO_CKPT_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../attribute_training/dino_classifiers_official.pth')
N_ATTRS_SUBSET = sum(len(v) for v in COLOR_SHAPE_MATERIAL_SUBSET.values())

from .predictor import CLIPTrainedAttributePredictor
from .predictor import DINOTrainedAttributePredictor