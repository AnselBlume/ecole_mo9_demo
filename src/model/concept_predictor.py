import torch
import numpy as np
import torch.nn as nn
from features import ImageFeatures
from weighted_predictor import WeightedPredictorOutput
from dataclasses import dataclass

class BatchedPredictor:
    pass

@dataclass
class ConceptPredictorOutput:
    img_score: torch.Tensor = None # (,)

    region_scores: torch.Tensor = None # (n_regions,)
    region_score: torch.Tensor = None # (,)

    trained_attr_img_scores: torch.Tensor = None # (n_trained_attrs,)
    trained_attr_region_scores: torch.Tensor = None # (n_regions, n_trained_attrs)

    zs_attr_img_scores: torch.Tensor = None # (n_zs_attrs,)
    zs_attr_region_scores: torch.Tensor = None # (n_regions, n_zs_attrs)

    all_scores_weighted: torch.Tensor = None # (1 + 1 + 2*n_trained_attrs + 2*n_zs_attrs,)
    cum_score: torch.Tensor = None # (,); For backpropagating loss or for prediction

class ConceptPredictor(nn.Module):
    # TODO incorporate unnamed visual features
    # TODO detect component concepts
    # TODO Add weightings for groups via FeatureGroups (e.g. necessary/descriptive, trained attr groups, img vs. region)

    def __init__(
        self,
        img_feature_dim: int,
        region_feature_dim: int,
        n_trained_attrs: int,
        n_zs_attrs: int,
        use_bias=True,
        use_ln=True,
        use_full_img=True
    ):
        super().__init__()

        self.img_feature_dim = img_feature_dim
        self.region_feature_dim = region_feature_dim
        self.n_trained_attrs = n_trained_attrs
        self.n_zs_attrs = n_zs_attrs
        self.use_bias = use_bias
        self.use_ln = use_ln

        # Modules
        self.img_predictor = nn.Sequential([
            nn.LayerNorm(img_feature_dim) if use_ln else nn.Identity(),
            nn.Linear(img_feature_dim, 1, bias=use_bias)
        ])

        self.region_predictor = nn.Sequential([
            nn.LayerNorm(region_feature_dim) if use_ln else nn.Identity(),
            nn.Linear(region_feature_dim, 1, bias=use_bias)
        ])

        self.full_img_scale = nn.Parameter(torch.randn(1)) if use_full_img else 0

        self.trained_attr_predictor = nn.Sequential([
            nn.LayerNorm(n_trained_attrs) if use_ln else nn.Identity(),
            nn.Linear(n_trained_attrs, n_trained_attrs, bias=use_bias)
        ])

        self.zs_attr_predictor = nn.Sequential([
            nn.LayerNorm(n_zs_attrs) if use_ln else nn.Identity(),
            nn.Linear(n_zs_attrs, n_zs_attrs, bias=use_bias)
        ])

        self.feature_groups = nn.ModuleDict()

    def forward(self, img_feats: ImageFeatures) -> ConceptPredictorOutput:
        if img_feats.all_scores is None: # If scores are not provided for feature_group-weighting, calculate them
            region_weights = img_feats.region_weights.unsqueeze(-1) # (n_regions, 1)

            img_score = self.img_predictor(img_feats.image_features) # (1, 1)
            img_score = img_score * self.full_img_scale # (1, 1)

            region_scores = self.region_predictor(img_feats.region_features) # (n_regions, 1)
            region_scores = region_scores * region_weights # (n_regions, 1)
            region_score = region_scores.sum(dim=1, keepdim=True) # (1, 1)

            trained_attr_img_scores = self.trained_attr_predictor(img_feats.trained_attr_img_scores) # (1, n_trained_attrs)
            trained_attr_img_scores = trained_attr_img_scores * self.full_img_scale # (1, n_trained_attrs)
            trained_attr_img_score = trained_attr_img_scores # (1, n_trained_attrs)

            trained_attr_region_scores = self.trained_attr_predictor(img_feats.trained_attr_region_scores) # (n_regions, n_trained_attrs)
            trained_attr_region_scores = trained_attr_region_scores * region_weights # (n_regions, n_trained_attrs)
            trained_attr_region_score = trained_attr_region_scores.sum(dim=0, keepdim=True) # (1, n_trained_attrs)

            zs_attr_img_scores = self.zs_attr_predictor(img_feats.zs_attr_img_scores.unsqueeze(0)) # (1, n_zs_attrs)
            zs_attr_img_scores = zs_attr_img_scores * self.full_img_scale # (1, n_zs_attrs)
            zs_attr_img_score = zs_attr_img_scores # (1, n_zs_attrs)

            zs_attr_region_scores = self.zs_attr_predictor(img_feats.zs_attr_region_scores) # (n_regions, n_zs_attrs)
            zs_attr_region_scores = zs_attr_region_scores * region_weights
            zs_attr_region_score = zs_attr_region_scores.sum(dim=0, keepdim=True) # (1, n_zs_attrs)

            all_scores = torch.cat([
                img_score, # (1, 1)
                region_score, # (1, 1)
                trained_attr_img_score, # (1, n_trained_attrs)
                trained_attr_region_score, # (1, n_trained_attrs)
                zs_attr_img_score, # (1, n_zs_attrs)
                zs_attr_region_score # (1, n_zs_attrs)
            ], dim=1) # (1, 1 + 1 + 2*n_trained_attrs + 2*n_zs_attrs)

        else:
            all_scores = img_feats.all_scores

        for group in self.feature_groups:
            all_scores = group(all_scores)

        # Construct output
        if img_feats.all_scores is None: # Intermediate results computed externally;
            output = ConceptPredictorOutput(
                all_scores_weighted=all_scores.detach().squeeze(),
                cum_score=all_scores.sum()
            )

        else: # Intermediate results computed here
            output = ConceptPredictorOutput(
                img_score=img_score.detach().squeeze(),
                region_scores=region_scores.detach().squeeze(),
                region_score=region_score.detach().squeeze(),
                trained_attr_img_scores=trained_attr_img_scores.detach().squeeze(),
                trained_attr_region_scores=trained_attr_region_scores.detach(),
                zs_attr_img_scores=zs_attr_img_scores.detach().squeeze(),
                zs_attr_region_scores=zs_attr_region_scores.detach(),
                all_scores_weighted=all_scores.detach().squeeze(),
                cum_score=all_scores.sum()
            )

        return output

    def add_zs_attr(self):
        # TODO better initialization from existing weights
        self.n_zs_attrs += 1
        self.zs_attr_predictor = nn.Linear(self.n_zs_attrs, self.n_zs_attrs, bias=self.use_bias)

    def remove_zs_attr(self):
        # TODO specify which attribute index to remove
        self.n_zs_attrs -= 1
        self.zs_attr_predictor = nn.Linear(self.n_zs_attrs, self.n_zs_attrs, bias=self.use_bias)
