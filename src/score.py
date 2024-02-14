import torch
from model.concept import ConceptDB, Concept
from PIL.Image import Image
from typing import Iterable, Union, Callable
from predictors.zero_shot_attrs import CLIPAttributePredictor

class AttributeScorer:
    def __init__(self, zs_predictor: CLIPAttributePredictor):
        self.zs_predictor = zs_predictor

    def _apply_weights(
        self,
        scores: torch.Tensor,
        img_weights: torch.Tensor = None,
        attr_weights: torch.Tensor = None
    ) -> torch.Tensor:
        '''
            scores: torch.Tensor of shape (n_imgs, n_attrs)
            img_weights: torch.Tensor of shape (n_imgs,). How much to weigh each image's attribute scores. Default is equal weighting.
            attr_weights: torch.Tensor of shape (n_attrs,). How much to weigh each attribute. Default is equal weighting.
        '''
        # Weight scores by image
        if img_weights is None:
            img_weights = torch.full((scores.shape[0],), fill_value=1/scores.shape[0])

        scores = scores * img_weights[:, None].to(scores.device) # (n_imgs, n_attrs)

        # Weights scores by attribute
        if attr_weights is None:
            attr_weights = torch.full((scores.shape[1],), fill_value=1/scores.shape[1])

        scores = scores * attr_weights[None, :].to(scores.device) # (n_imgs, n_attrs)

        return scores

    def zs_scores(
        self,
        imgs: Iterable[Image],
        texts: Iterable[str]
    ) -> torch.Tensor:
        '''
            Returns a torch.Tensor of matching scores with shape (n_imgs, n_texts)
        '''
        return self.zs_predictor.predict(imgs, texts)

    @torch.inference_mode()
    def score_regions(
        self,
        regions: Iterable[Image],
        zs_texts: Iterable[str],
        region_weights: torch.Tensor = None,
        zs_text_weights: torch.Tensor = None,
        zs_weight: float = 1.
    ) -> torch.Tensor:
        '''
            Returns: torch.Tensor of shape (n_regions, n_texts), matching scores between each region and each text.
        '''
        # TODO add support for learned attributes
        ret_dict = {}

        # Zero-shot scores
        raw_zs_scores = self.zs_scores(regions, zs_texts) # (n_regions, n_texts)
        ret_dict['zs_scores_per_region_raw'] = raw_zs_scores

        weighted_zs_scores = self._apply_weights(raw_zs_scores, img_weights=region_weights, attr_weights=zs_text_weights) # (n_regions, n_texts)
        ret_dict['zs_scores_per_region_weighted'] = weighted_zs_scores * zs_weight

        return ret_dict

class ConceptScorer:
    def __init__(self):
        # TODO use AttributeScorer
        pass

    # TODO Batched computation of scores via padding or NestedTensors?
    def zs_attr_scores(self, img: Image, concepts: Iterable[Concept]) -> torch.Tensor:
        pass


    def learned_attr_scores(self, img: Image, concepts: Iterable[Concept]) -> torch.Tensor:
        pass

    def score(self, img: Image, concepts: Iterable[Concept]) -> torch.Tensor:
        '''
            Returns: torch.Tensor of shape (n_concepts,), image scores for each provided concept.

             TODO Handle necessary vs descriptive attributes

             TODO Handle component concepts

             TODO Handle direct detection of concepts. This can occur with a learned binary detector, or with a zero-shot detector,
             this would necessitate checking the concept's sibling concepts, and if the scores are similar enough, then not incorporating
             the score

             TODO Handle unnamed visual attributes

             TODO Possibly do something with occlusion generation?

        '''
        zs_scores = self.zs_attr_scores(img, concepts)
        learned_scores = self.learned_attr_scores(img, concepts)

        # Each concept weighs the scores differently
        zs_weights = torch.tensor([c.zs_attr_score_weight for c in concepts])
        learned_attr_weights = torch.tensor([c.learned_attr_score_weight for c in concepts])

        zs_scores = zs_scores * zs_weights
        learned_scores = learned_scores * learned_attr_weights

        # Compute final scores
        scores = zs_scores + learned_scores

        return scores

    def score_regions(
        self,
        regions: Iterable[Image],
        concepts: Iterable[Concept] = None,
        region_weights: torch.Tensor = None
    ):
        '''
            Returns: torch.Tensor of shape (n_regions, n_concepts), image scores of each region for each concept in
            the concepts iterable
        '''
        if region_weights is None: # Equally weight regions
            region_weights = torch.full((len(regions),), fill_value=1/len(regions))

        scores = [self.score(region, concepts) for region in regions] # TODO batched scoring?
        scores = torch.stack(scores, dim=0) # (n_regions, n_concepts)
        scores = scores * region_weights[:, None] # (n_regions, n_concepts)

        return scores

    def score_image_from_regions(self, regions: list[Image], concepts: Iterable[Concept], region_weights: torch.Tensor = None):
        '''
            Returns: torch.Tensor of shape (n_concepts,), image scores for each concept in the concept set.
        '''
        scores = self.score_regions(regions, concepts, region_weights)
        scores = scores.sum(dim=0)

        return scores