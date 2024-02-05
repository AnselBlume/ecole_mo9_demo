import torch
from concept import ConceptSet
from PIL.Image import Image

class Scorer:
    # TODO Batched computation of scores via padding or NestedTensors?
    def zs_attr_scores(self, img: Image, concept_set: ConceptSet) -> torch.Tensor:
        pass

    def learned_attr_scores(self, img: Image, concept_set: ConceptSet) -> torch.Tensor:
        pass

    def score(self, img: Image, concept_set: ConceptSet) -> torch.Tensor:
        '''
            Returns: torch.Tensor of shape (n_concepts,), image scores for each concept in the concept set.

             TODO Handle necessary vs descriptive attributes

             TODO Handle component concepts

             TODO Handle direct detection of concepts. This can occur with a learned binary detector, or with a zero-shot detector,
             this would necessitate checking the concept's sibling concepts, and if the scores are similar enough, then not incorporating
             the score

        '''
        zs_scores = self.zs_attr_scores(img, concept_set)
        learned_scores = self.learned_attr_scores(img, concept_set)

        # Each concept weighs the scores differently
        zs_weights = torch.tensor([c.zs_attr_score_weight for c in concept_set])
        learned_attr_weights = torch.tensor([c.learned_attr_score_weight for c in concept_set])

        zs_scores = zs_scores * zs_weights
        learned_scores = learned_scores * learned_attr_weights

        # Compute final scores
        scores = zs_scores + learned_scores

        return scores

    def score_regions(self, regions: list[Image], region_weights: torch.Tensor, concept_set: ConceptSet):
        '''
            Returns: torch.Tensor of shape (n_regions, n_concepts), image scores of each region for each concept in
            the concept set.
        '''
        scores = [self.score(region, concept_set) for region in regions] # TODO batched scoring?
        scores = torch.stack(scores, dim=0) # (n_regions, n_concepts)
        scores = scores * region_weights[:, None] # (n_regions, n_concepts)

        return scores

    def score_image_from_regions(self, regions: list[Image], region_weights: torch.Tensor, concept_set: ConceptSet):
        '''
            Returns: torch.Tensor of shape (n_concepts,), image scores for each concept in the concept set.
        '''
        scores = self.score_regions(regions, region_weights, concept_set)
        scores = scores.sum(dim=0)

        return scores