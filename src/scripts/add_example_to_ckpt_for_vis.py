'''
    Script to add an example to a ConceptKB checkpoint for visualization purposes.
    Namely, adds it to the start of a concept's example list, so that it is selected as the exemplar for concept-concept comparison.
    Backs up the original checkpoint, adds the example, caches the segmentations and features, and saves the modified checkpoint to the original path.
'''
# %%
import os
import shutil
import logging
import coloredlogs
os.chdir('/shared/nas2/blume5/fa23/ecole/src/mo9_demo/src')

from kb_ops.caching import ConceptKBFeatureCacher
from model.concept import ConceptKB, ConceptExample
from scripts.utils import get_timestr
from feature_extraction import build_feature_extractor, build_sam
from image_processing import build_localizer_and_segmenter
from kb_ops.feature_pipeline import ConceptKBFeaturePipeline

logger = logging.getLogger(__file__)
coloredlogs.install(level='DEBUG', logger=logger)

# %% Build ConceptKB and cacher
ckpt_path = '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_06_06-23:31:12-8ckp59v8-all_planes_and_guns/concept_kb_epoch_50.pt'

ckpt_backup_path = ckpt_path + '.bak-' + get_timestr()
shutil.copy(ckpt_path, ckpt_backup_path)
logging.info(f'Backed up the checkpoint to {ckpt_backup_path}')

logger.info('Loading checkpoint...')
concept_kb = ConceptKB.load(ckpt_path)

# Build cacher
loc_and_seg = build_localizer_and_segmenter(build_sam(), None)
fe = build_feature_extractor()
feature_pipeline = ConceptKBFeaturePipeline(loc_and_seg, fe)

cache_root = '/shared/nas2/blume5/fa23/ecole/modified_ckpt_feature_cache'
cacher = ConceptKBFeatureCacher(concept_kb, feature_pipeline, cache_dir='/shared/nas2/blume5/fa23/ecole/modified_ckpt_feature_cache')

# %% Add a new example
new_image_path = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/test_selected/known class/sniper rifle 1.png'
concept_name = 'sniper rifle'

new_example = ConceptExample(concept_name=concept_name, image_path=new_image_path)

concept_kb[concept_name].examples.insert(0, new_example)

# %% Cache the example's segmentations and features
cacher.cache_segmentations()
cacher.cache_features()

# %% Save the modified checkpoint
logger.info(f'Saving modified checkpoint to original checkpoint path: {ckpt_path}')
concept_kb.save(ckpt_path)
# %%
