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
    img_score: torch.Tensor # (1,)
    region_scores: torch.Tensor # (n_regions, 1)
    region_score: torch.Tensor # (1,)
    learned_attr_scores: torch.Tensor # (n_learned_attrs,)
    zs_attr_scores: torch.Tensor # (n_zs_attrs,)
    cum_score: torch.Tensor # For backpropagating loss or prediction

class ConceptPredictor(nn.Module):
    # TODO incorporate unnamed visual features
    # TODO detect component concepts
    # TODO Add weightings for groups via feature_groups (e.g. necessary/descriptive)

    def __init__(
        self,
        img_feature_dim: int,
        region_feature_dim: int,
        n_learned_attrs: int,
        n_zs_attrs: int,
        use_bias=True,
        use_ln=True
    ):
        super().__init__()

        self.img_feature_dim = img_feature_dim
        self.region_feature_dim = region_feature_dim
        self.n_learned_attrs = n_learned_attrs
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

        self.learned_attr_predictor = nn.Sequential([
            nn.LayerNorm(n_learned_attrs) if use_ln else nn.Identity(),
            nn.Linear(n_learned_attrs, n_learned_attrs, bias=use_bias)
        ])

        self.zs_attr_predictor = nn.Sequential([
            nn.LayerNorm(n_zs_attrs) if use_ln else nn.Identity(),
            nn.Linear(n_zs_attrs, n_zs_attrs, bias=use_bias)
        ])

        self.feature_groups = nn.ModuleDict()

    def forward(self, img_feats: ImageFeatures) -> ConceptPredictorOutput:
        if img_feats.all_scores is None: # If scores are not provided for feature_group_weighting, calculate them
            img_score = self.img_predictor(img_feats.image_features.unsqueeze(0)) # (1, 1)

            region_scores = self.region_predictor(img_feats.region_features) # (n_regions, 1)
            region_scores = region_scores.T * img_feats.region_feature_weights # (1, n_regions)
            region_score = region_scores.sum(dim=1, keepdim=True) # (1, 1)

            learned_attr_scores = self.learned_attr_predictor(img_feats.learned_attr_scores.unsqueeze(0)) # (1, n_learned_attrs)
            zs_attr_scores = self.zs_attr_predictor(img_feats.zs_attr_scores.unsqueeze(0)) # (1, n_zs_attrs)

            all_scores = torch.cat([img_score, region_score, learned_attr_scores, zs_attr_scores], dim=1) # (1, n_features)

        else:
            if img_feats.region_scores is None:
                raise ValueError('region_scores must be provided when all_scores is provided.')

            region_scores = img_feats.region_scores
            all_scores = img_feats.all_scores

        for group in self.feature_groups:
            all_scores = group(all_scores)

        # Construct output
        output = ConceptPredictorOutput()
        offset = 0

        output.img_score = all_scores[0, offset].detach()
        offset += 1

        output.region_scores = region_scores.detach()
        output.region_score = all_scores[0, offset].detach()
        offset += 1

        output.learned_attr_scores = all_scores[0, offset:offset+self.n_learned_attrs].detach()
        offset += self.n_learned_attrs

        output.zs_attr_scores = all_scores[0, offset:offset+self.n_zs_attrs].detach()

        output.cum_score = all_scores.sum()

        return output

    def add_zs_attr(self):
        # TODO better initialization from existing weights
        self.n_zs_attrs += 1
        self.zs_attr_predictor = nn.Linear(self.n_zs_attrs, self.n_zs_attrs, bias=self.use_bias)

    def remove_zs_attr(self):
        # TODO specify which attribute index to remove
        self.n_zs_attrs -= 1
        self.zs_attr_predictor = nn.Linear(self.n_zs_attrs, self.n_zs_attrs, bias=self.use_bias)
