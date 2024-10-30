# %%
import logging

import os
import sys
sys.path = [os.path.realpath(os.path.join(__file__, '../..'))] + sys.path

# Cannot use NFS with mp temp files as otherwise get "device or resource busy" errors
tmp_dir = '/scratch/tmp'
os.makedirs(tmp_dir, exist_ok=True)
os.environ['TMPDIR'] = tmp_dir

from controller import Controller

logger = logging.getLogger(__file__)

if __name__ == '__main__':
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    import PIL
    from feature_extraction import build_feature_extractor, build_sam, build_desco
    from image_processing import build_localizer_and_segmenter
    from model.concept import ConceptKB, ConceptExample
    import PIL.Image
    import coloredlogs
    from kb_ops.feature_pipeline import ConceptKBFeaturePipeline
    import sys
    from kb_ops.concurrency import LockType

    coloredlogs.install(level=logging.INFO)

    # %%
    ###############################
    #  June 2024 Demo Checkpoint #
    ###############################
    # ckpt_path = '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_06_06-23:31:12-8ckp59v8-all_planes_and_guns/concept_kb_epoch_50.pt' # Default checkpoint
    # ckpt_path = '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_07_23-00:16:09-all_planes_and_guns_v3-rm_bg/concept_kb_epoch_21.pt' # With infer localize
    # ckpt_path = '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_07_16-01:37:06-all_guns_2/concept_kb_epoch_10.pt'
    # ckpt_path = '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_07_23-02:04:37-all_planes_and_guns_v3-rm_bg_with_component_rem_bg/concept_kb_epoch_129.pt'

    ckpt_path = '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_07_23-02:35:55-all_planes_and_guns_v3-rm_bg_with_component_rm_bg_containing_positives/concept_kb_epoch_400.pt' # Close to actual chekcpoint

    kb = ConceptKB.load(ckpt_path)
    loc_and_seg = build_localizer_and_segmenter(build_sam(), None)
    fe = build_feature_extractor()
    feature_pipeline = ConceptKBFeaturePipeline(loc_and_seg, fe)
    controller = Controller(kb, feature_pipeline)
    file_lock_type = LockType.FILE_LOCK
    train_devices = [0, 1, 2, 3]
    train_n_epochs = 100

    controller.compare_component_concepts('airplane', 'gun')
    img = PIL.Image.open('/shared/nas2/blume5/fa23/ecole/biplane 1.jpg')
    heatmap, detection_score = controller.heatmap(img, 'afterburner', return_detection_score=True)
    heatmap.save('/shared/nas2/blume5/fa23/ecole/heatmap.jpg')

    new_concept = controller.add_concept('spoon')
    new_concept.examples = [
        ConceptExample('spoon', image_path='/shared/nas2/blume5/fa23/ecole/spoons/spoon_1.jpg'),
        ConceptExample('spoon', image_path='/shared/nas2/blume5/fa23/ecole/spoons/spoon_2.jpg'),
        ConceptExample('spoon', image_path='/shared/nas2/blume5/fa23/ecole/spoons/spoon_6_aug_2.jpg')
    ]

    controller.train_concepts_parallel(['spoon'], devices=train_devices, n_epochs=train_n_epochs, lock_type=file_lock_type)

    new_concept = controller.add_concept('fork')
    new_concept.examples = [
        ConceptExample('fork', image_path='/shared/nas2/blume5/fa23/ecole/forks/fork_4.jpg'),
        ConceptExample('fork', image_path='/shared/nas2/blume5/fa23/ecole/forks/fork_5_aug_2.jpg'),
        ConceptExample('fork', image_path='/shared/nas2/blume5/fa23/ecole/forks/fork_9_aug_2.jpg')
    ]
    controller.train_concepts_parallel(['fork'], devices=train_devices, n_epochs=train_n_epochs, lock_type=file_lock_type)

    spoon_and_fork_img_path = '/shared/nas2/blume5/fa23/ecole/spoon and fork.jpg'
    spoon_and_fork_img = PIL.Image.open(spoon_and_fork_img_path).convert('RGB')
    # %%
    heatmap = controller.heatmap_class_intersection('fork', 'spoon', spoon_and_fork_img)
    # heatmap.save('/shared/nas2/blume5/fa23/ecole/heatmap.jpg')

    # new_concept = controller.add_concept('handle', containing_concept_names=['spoon'])
    # new_concept.examples = [
    #     ConceptExample('handle', image_path='/shared/nas2/blume5/fa23/ecole/handle/fork_handle.jpg'),
    #     ConceptExample('handle', image_path='/shared/nas2/blume5/fa23/ecole/handle/spoon_handle.png'),
    #     ConceptExample('handle', image_path='/shared/nas2/blume5/fa23/ecole/handle/knife_handle.png')
    # ]

    # controller.train_concepts_parallel(['handle'], devices=[0, 1, 2, 3], n_epochs=100, lock_type=file_lock_type)
    print('Done')
    # sys.exit(0)
    # %%
    # test_path = '/shared/nas2/blume5/fa23/ecole/forks/test_fork.jpg'
    # prediction = controller.predict_hierarchical(PIL.Image.open(test_path).convert('RGB'))
    # print(f'Predicted label for fork image: {prediction["predicted_label"]}')

    # test_path = '/shared/nas2/blume5/fa23/ecole/spoons/test_spoon.png'
    # prediction = controller.predict_hierarchical(PIL.Image.open(test_path).convert('RGB'))
    # print(f'Predicted label for spoon image: {prediction["predicted_label"]}')
    # sys.exit(0)

    # new_concept = controller.add_concept('malamute')
    # malamute_dir = '/shared/nas2/blume5/fa23/ecole/malamute/train'
    # new_concept.examples = [
    #     ConceptExample('malamute', image_path=os.path.join(malamute_dir, p))
    #     for p in os.listdir(malamute_dir)
    # ]
    # controller.train_concepts_parallel(['malamute'], devices=[4, 5, 6, 7], n_epochs=100, lock_type=file_lock_type)

    # new_concept = controller.add_concept('husky')
    # husky_dir = '/shared/nas2/blume5/fa23/ecole/husky/train'
    # new_concept.examples = [
    #     ConceptExample('husky', image_path=os.path.join(husky_dir, p))
    #     for p in os.listdir(husky_dir)
    # ]
    # controller.train_concepts_parallel(['husky'], devices=[0, 1, 2, 3], n_epochs=100, lock_type=file_lock_type)

    # test_path = '/shared/nas2/blume5/fa23/ecole/malamute/test/69f757a798b01de2e8662fcf81972b90.jpg'
    # prediction = controller.predict_hierarchical(PIL.Image.open(test_path).convert('RGB'))
    # print(f'Predicted label for malamute image: {prediction["predicted_label"]}')

    # test_path = '/shared/nas2/blume5/fa23/ecole/husky/test/siberian-husky-dog-breed-info.jpeg'
    # prediction = controller.predict_hierarchical(PIL.Image.open(test_path).convert('RGB'))
    # print(f'Predicted label for husky image: {prediction["predicted_label"]}')

    # controller.train_concepts_parallel(['passenger plane', 'biplane', 'row of windows'], devices=[0, 1, 2, 3])

    # Sniper Heatmap Generation
    img_path = '/shared/nas2/blume5/fa23/ecole/1-biplane.jpg'
    img = PIL.Image.open(img_path).convert('RGB')
    heatmap = controller.heatmap(img, 'biplane')
    heatmap.save('/shared/nas2/blume5/fa23/ecole/heatmap.jpg')

    # %% Add a new concept and train it
    new_concept_name = 'bomber'
    controller.add_concept(new_concept_name, parent_concept_names=['airplane'])

    image_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/bomber'

    examples = [
        ConceptExample(concept_name=new_concept_name, image_path=os.path.join(image_dir, basename))
        for basename in os.listdir(image_dir)
    ]

    controller.add_examples(examples, new_concept_name)
    controller.train_concept(new_concept_name)

    # Add another new concept
    new_concept_name = 'bomber wing'
    # controller.add_concept(new_concept_name, parent_concept_names=['bomber'])
    controller.add_concept(new_concept_name, containing_concept_names=['bomber'])

    image_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/bomber_wings'

    examples = [
        ConceptExample(concept_name=new_concept_name, image_path=os.path.join(image_dir, basename))
        for basename in os.listdir(image_dir)
    ]

    controller.add_examples(examples, new_concept_name)
    controller.train_concepts(['bomber']) # XXX This has a caching bug; must be fixed!

    # %% Run the first prediction
    img_path = '/shared/nas2/blume5/fa23/ecole/Screenshot 2024-06-07 at 6.23.02 AM.png'
    img = PIL.Image.open(img_path).convert('RGB')
    result = controller.is_concept_in_image(img, 'wing-mounted engine', unk_threshold=.001)

    # %%
    result = controller.predict_concept(img, unk_threshold=.1)
    logger.info(f'Predicted label: {result["predicted_label"]}')

    # %% Hierarchical prediction
    result = controller.predict_hierarchical(img, unk_threshold=.1)
    logger.info(f'Concept path: {result["concept_path"]}')

    # %% New concept training
    new_concept_name = 'biplane'
    parent_concept_name = 'airplane'

    new_concept_image_paths = [
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/biplane/000001.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/biplane/000002.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/biplane/000003.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/biplane/000004.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/biplane/000005.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/biplane/000006.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/biplane/000008.jpg'
    ]

    new_concept_examples = [ConceptExample(new_concept_name, image_path=image_path) for image_path in new_concept_image_paths]

    controller.add_concept(new_concept_name, parent_concept_names=[parent_concept_name])

    controller.train_concept(
        new_concept_name,
        new_examples=new_concept_examples
    )

    # %% Predict with new concept
    test_image_paths = [
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/biplane/000010.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/biplane/000011.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/passenger plane/000021.jpg',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/fighter jet/Screenshot 2024-06-02 at 9.18.58 PM.png',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/afterburner/sr71.png',
        '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/june_demo_2024/airplanes_v2/row of windows/000021.jpg'
    ]
    test_image_labels = [
        'biplane',
        'biplane',
        'passenger plane',
        'fighter jet',
        'afterburner',
        'row of windows'
    ]
    include_component_parts = [
        False,
        False,
        False,
        False,
        True,
        True
    ]

    for img_path, label, include_components in zip(test_image_paths, test_image_labels, include_component_parts):
        img = PIL.Image.open(img_path).convert('RGB')
        result = controller.predict_concept(img, unk_threshold=.1, include_component_concepts=include_components)
        logger.info(f'Predicted label: {result["predicted_label"]}')

        result = controller.predict_hierarchical(img, unk_threshold=.1, include_component_concepts=include_components)
        logger.info(f'Hierarchical Predicted label: {result["predicted_label"]}')

        logger.info(f'Expected label: {label}')

    # %%
    ###############################
    #  March 2024 Demo Checkpoint #
    ###############################
    ckpt_path = '/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_03_22-15:06:03-xob6535d-v3-dino_pool/concept_kb_epoch_50.pt'

    kb = ConceptKB.load(ckpt_path)
    loc_and_seg = build_localizer_and_segmenter(build_sam(), build_desco())
    fe = build_feature_extractor()
    feature_pipeline = ConceptKBFeaturePipeline(loc_and_seg, fe)

    controller = Controller(kb, feature_pipeline)

    # %% Run the first prediction
    img_path = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/adversarial_spoon.jpg'
    img = PIL.Image.open(img_path).convert('RGB')
    result = controller.predict_concept(img, unk_threshold=.1)

    logger.info(f'Predicted label: {result["predicted_label"]}')

    # %% Run the second prediction
    img_path2 = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/assets/fork_2_9.jpg'
    img2 = PIL.Image.open(img_path2).convert('RGB')
    controller.predict_concept(img2)

    # %% Explain difference between images
    logger.info('Explaining difference between predictions...')
    controller.compare_predictions(indices=(-2,-1), weight_by_predictors=True)

    # %% Explain difference between image regions
    controller.compare_predictions(indices=(-2,-1), weight_by_predictors=True, image1_regions=[0])
    controller.compare_predictions(indices=(-2,-1), weight_by_predictors=True, image2_regions=[0])
    controller.compare_predictions(indices=(-2,-1), weight_by_predictors=True, image1_regions=[0], image2_regions=[0])

    # %% Explain difference between concepts
    controller.compare_concepts('spoon', 'fork')

    # %% Visualize difference between zero-shot attributes
    controller.compare_zs_attributes(('spoon', 'fork'), img)