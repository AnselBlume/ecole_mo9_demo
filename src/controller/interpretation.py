from .base import BaseController
from model.concept import Concept
import torch
from typing import Union, Literal
from PIL.Image import Image
from visualization.vis_utils import  plot_image_differences, plot_concept_differences, plot_zs_attr_differences

class ControllerInterpretationMixin(BaseController):
    ##################
    # Interpretation #
    ##################
    def compare_concepts(self, concept_name1: str, concept_name2: str, weight_by_magnitudes: bool = True):
        concept1 = self.retrieve_concept(concept_name1)
        concept2 = self.retrieve_concept(concept_name2)

        attr_names = self.feature_pipeline.feature_extractor.trained_attr_predictor.attr_names

        return plot_concept_differences(
            (concept1, concept2),
            attr_names,
            weight_by_magnitudes=weight_by_magnitudes,
            take_abs_of_weights=self.concept_kb.cfg.use_ln,
            return_img=True
        )

    def compare_component_concepts(self, concept_name1: str, concept_name2: str) -> tuple[list[str], list[str]]:
        '''
            Returns the names of the component concepts exclusive to each of the two concepts.

            Reasoning:
            If you have concepts A, B, you don't do Parts(A) \ Parts(B) for the parts exclusive to A; instead, you do Parts(A) \ Parts(B) U Ancestors(Parts(B)).

            So if concept A == Biplane and B == Airplane, then Propeller is a difference between a Biplane and an Airplane because biplanes have propellers
            but Airplanes don't necessarily (they just have generic propulsion components).

            However, a propulsion component is not a difference between an Airplane and a Biplane, because Airplanes have propulsion components and so do Biplanes
            (since a propeller is a type—descendant—of propulsion component)
        '''


        concept1 = self.retrieve_concept(concept_name1)
        concept2 = self.retrieve_concept(concept_name2)

        # Get all ancestors of a concept's parts (including the parts themselves)
        def get_part_ancestor_union(concept: Concept) -> set[Concept]:
            part_ancestor_union = set()

            for part in concept.component_concepts.values():
                ancestors = self.concept_kb.rooted_subtree(part, reverse_edges=True)
                part_ancestor_union.update(ancestors)

            return part_ancestor_union

        concept1_part_union = get_part_ancestor_union(concept1)
        concept2_part_union = get_part_ancestor_union(concept2)

        # Take difference of part unions to get parts exclusive to each concept
        concept1_parts = concept1_part_union - concept2_part_union
        concept2_parts = concept2_part_union - concept1_part_union

        concept1_part_names = sorted([p.name for p in concept1_parts])
        concept2_part_names = sorted([p.name for p in concept2_parts])

        return concept1_part_names, concept2_part_names

    def compare_predictions(
        self,
        indices: tuple[int,int] = None,
        images: tuple[Image,Image] = None,
        weight_by_predictors: bool = True,
        image1_regions: Union[str, list[int]] = None,
        image2_regions: Union[list[str], list[int]] = None
    ) -> Image:
        if not ((images is None) ^ (indices is None)):
            raise ValueError('Exactly one of imgs or idxs must be provided.')

        if images is not None:
            if len(images) != 2:
                raise ValueError('imgs must be a tuple of length 2.')

            # Cache predictions
            # NOTE This can be optimized, since running through all concept predictors when only need to
            # 1) Localize and segment
            # 2) Compute image features
            # 3) Compute trained attribute predictions
            self.predict_concept(images[0], unk_threshold=0)
            self.predict_concept(images[1], unk_threshold=0)

            indices = (-2, -1)

        else: # idxs is not None
            if len(indices) != 2:
                raise ValueError('idxs must be a tuple of length 2.')

            images = (self.cached_images[indices[0]], self.cached_images[indices[1]])

        predictions1 = self.cached_predictions[indices[0]]
        predictions2 = self.cached_predictions[indices[1]]

        # Handle case where user wants to visualize region predictions
        if image1_regions or image2_regions:
            def get_region_inds(regions: Union[str, list[int]], predictions: dict):
                part_names = predictions['segmentations'].part_names

                if isinstance(regions, str):
                    inds = [i for i, part_name in enumerate(part_names) if part_name == regions]
                else:
                    inds = regions

                return inds

            def get_region_scores(region_inds: list[int], predictions: dict):
                region_scores = predictions['predicted_concept_outputs'].trained_attr_region_scores[region_inds]
                return region_scores.mean(dim=0) # Average over number of regions

        # Get attr scores and masks for regions or image
        if image1_regions:
            region_inds = get_region_inds(image1_regions, predictions1)
            trained_attr_scores1 = get_region_scores(region_inds, predictions1)
            region_mask1 = predictions1['segmentations'].part_masks[region_inds]
        else:
            trained_attr_scores1 = predictions1['predicted_concept_outputs'].trained_attr_img_scores
            region_mask1 = None

        if image2_regions:
            region_inds = get_region_inds(image2_regions, predictions1)
            trained_attr_scores2 = get_region_scores(region_inds, predictions2)
            region_mask2 = predictions2['segmentations'].part_masks[region_inds]
        else:
            trained_attr_scores2 = predictions2['predicted_concept_outputs'].trained_attr_img_scores
            region_mask2 = None

        # Convert to probabilities if not already
        if not self.concept_kb.cfg.use_probabilities:
            trained_attr_scores1 = trained_attr_scores1.sigmoid()
            trained_attr_scores2 = trained_attr_scores2.sigmoid()

        # Get attribute names
        attr_names = self.feature_pipeline.feature_extractor.trained_attr_predictor.attr_names

        # Extract winning predictors if weighting by predictors
        if weight_by_predictors:
            predicted_concept1 = predictions1['concept_names'][predictions1['predicted_index']]
            predicted_concept2 = predictions2['concept_names'][predictions2['predicted_index']]

            predictor1 = self.concept_kb[predicted_concept1].predictor
            predictor2 = self.concept_kb[predicted_concept2].predictor
            predictors = (predictor1, predictor2)
        else:
            predictors = ()

        return plot_image_differences(
            images,
            (trained_attr_scores1, trained_attr_scores2),
            attr_names,
            weight_imgs_by_predictors=predictors,
            region_masks=(region_mask1, region_mask2),
            return_img=True,
        )

    def compare_zs_attributes(
        self,
        concept_names: tuple[str,str],
        image: Image,
        use_sigmoid: bool = False,
        weight_scores_by_predictors: bool = False
    ):
        if len(concept_names) != 2:
            raise ValueError('concepts must be a tuple of length 2.')

        concept1 = self.retrieve_concept(concept_names[0])
        concept2 = self.retrieve_concept(concept_names[1])

        concept1_zs_attr_names = [a.name for a in concept1.zs_attributes]
        concept2_zs_attr_names = [a.name for a in concept2.zs_attributes]
        all_queries = [a.query for a in concept1.zs_attributes + concept2.zs_attributes]

        # Forward pass through CLIP
        scores = self.feature_pipeline.feature_extractor.zs_attr_predictor.predict([image], all_queries, apply_sigmoid=use_sigmoid) # (1, n_zs_attrs)

        if use_sigmoid:
            scores = scores.sigmoid()

        concept1_scores, concept2_scores = scores[0].split((len(concept1.zs_attributes), len(concept2.zs_attributes)))

        if weight_scores_by_predictors:
            concept1_weights = concept1.predictor.zs_attr_predictor.weight.data.cpu()
            concept2_weights = concept2.predictor.zs_attr_predictor.weight.data.cpu()
            predictor_weights = concept1_weights, concept2_weights

        else:
            predictor_weights = ()

        return plot_zs_attr_differences(
            image,
            zs_attr_names=(concept1_zs_attr_names, concept2_zs_attr_names),
            concept_names=(concept1.name, concept2.name),
            concept_scores=(concept1_scores, concept2_scores),
            weight_scores_by_predictors=predictor_weights
        )

    def get_maximizing_region(
        self,
        index: int,
        attr_name: str,
        attr_type: Literal['trained', 'zs'] = 'trained',
        use_abs: bool = False,
        return_all_metadata: bool = False
    ) -> Union[torch.BoolTensor, dict]:
        '''
            Gets the part mask corresponding to the region with the highest score for the
            specified attribute. Assumes the prediction has already been cached.

            Arguments:
                index: Index of the cached prediction to use.
                attr_name: Name of the attribute to visualize.
                attr_type: Type of attribute to visualize. Must be 'trained' or 'zs'.
                use_abs: If True, finds the region which maximizes the (unsigned) magnitude of the score.
                return_all_metadata: If True, returns a dict with keys 'part_masks', 'maximizing_index',
                    and 'attr_scores_by_region'. If False, returns the part mask BoolTensor.

            Returns: If return_all_metadata is False, returns the part mask. Otherwise, returns a dict
                with keys 'part_masks', 'maximizing_index', and 'attr_scores_by_region'.

        '''
        prediction = self.cached_predictions[index]

        if attr_type == 'trained':
            attr_names = self.feature_pipeline.feature_extractor.trained_attr_predictor.attr_names
            attr_scores = prediction['predicted_concept_outputs'].trained_attr_region_scores

        else:
            assert attr_type == 'zs'
            attr_names = [a.name for a in self.concept_kb[prediction['predicted_label']].zs_attributes]
            attr_scores = prediction['predicted_concept_outputs'].zs_attr_region_scores

        attr_index = attr_names.index(attr_name)
        attr_scores_by_region = attr_scores[:, attr_index] # (n_regions,)

        if use_abs: # Find region which maximizes magnitude of score, regardless of sign
            attr_scores_by_region = attr_scores_by_region.abs()

        maximizing_region_ind = attr_scores_by_region.argmax().item()

        if return_all_metadata:
            return {
                'part_masks': prediction['segmentations'].part_masks,
                'maximizing_index': maximizing_region_ind,
                'attr_scores_by_region': attr_scores_by_region
            }

        else:
            maximizing_region_mask = prediction['segmentations'].part_masks[maximizing_region_ind]
            return maximizing_region_mask

    def explain_prediction(self, index: int = -1):
        # See scripts/vis_contributions.py
        pass

