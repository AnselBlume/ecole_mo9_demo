import torch
import torch.nn as nn
from model.features import ImageFeatures
from dataclasses import dataclass
from model.dataclass_base import DeviceShiftable

class BatchedPredictor:
    pass

@dataclass
class ConceptPredictorFeatures(ImageFeatures):
    zs_attr_img_scores: torch.Tensor = None # (1, n_zs_attrs)
    zs_attr_region_scores: torch.Tensor = None # (n_regions, n_zs_attrs)

    component_concept_scores: torch.Tensor = None # (1, n_component_concepts)

@dataclass
class ConceptPredictorOutput(DeviceShiftable):
    img_score: torch.Tensor = None # (,)

    region_scores: torch.Tensor = None # (n_regions,)
    region_score: torch.Tensor = None # (,)

    trained_attr_img_scores: torch.Tensor = None # (n_trained_attrs,)
    trained_attr_region_scores: torch.Tensor = None # (n_regions, n_trained_attrs)

    zs_attr_img_scores: torch.Tensor = None # (n_zs_attrs,)
    zs_attr_region_scores: torch.Tensor = None # (n_regions, n_zs_attrs)

    component_concept_scores: torch.Tensor = None # (n_component_concepts,)

    all_scores_weighted: torch.Tensor = None # (1 + 1 + 2*n_trained_attrs + 2*n_zs_attrs,)
    cum_score: torch.Tensor = None # (,); For backpropagating loss or for prediction

class Hadamard(nn.Module):
    def __init__(self, n_features: int, bias: bool = False):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(n_features))
        self.bias = nn.Parameter(torch.randn(n_features)) if bias else None

    def forward(self, x: torch.Tensor):
        x = x * self.weights

        if self.bias is not None:
            x = x + self.bias

        return x

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
        n_component_concepts: int = 0,
        use_bias=True,
        use_ln=True,
        use_probabilities=False,
        use_full_img=True,
        use_regions=True,
        use_region_features=True
    ):
        super().__init__()

        self.img_feature_dim = img_feature_dim
        self.region_feature_dim = region_feature_dim
        self.n_trained_attrs = n_trained_attrs
        self.n_zs_attrs = n_zs_attrs
        self.n_component_concepts = n_component_concepts
        self.use_bias = use_bias
        self.use_ln = use_ln
        self.use_full_img = use_full_img
        self.use_regions = use_regions
        self.use_region_features = use_region_features

        # Modules
        self.n_features = (
            use_full_img + use_regions
            + (use_full_img + use_regions) * n_trained_attrs
            + (use_full_img + use_regions) * n_zs_attrs
            + n_component_concepts
        )

        if use_ln and use_probabilities:
            raise ValueError("Layer normalization and probabilities cannot be used together")

        self.ln = nn.LayerNorm(self.n_features) if use_ln else nn.Identity()
        self.prob_scaler = nn.Sigmoid() if use_probabilities else nn.Identity()

        self.img_features_predictor = nn.Linear(img_feature_dim, 1, bias=use_bias) # This will only be used if use_full_img is True
        self.img_features_weight = nn.Linear(1, 1, bias=use_bias) if self.use_full_img else nn.Identity()

        self.region_features_predictor = nn.Linear(region_feature_dim, 1, bias=use_bias) # This will only be used if use_regions is True
        self.region_features_weight = nn.Linear(1, 1, bias=use_bias) if self.use_regions and self.use_region_features else nn.Identity()

        self.img_trained_attr_weights = Hadamard(n_trained_attrs, bias=use_bias) if n_trained_attrs else nn.Identity()
        self.regions_trained_attr_weights = Hadamard(n_trained_attrs, bias=use_bias) if n_trained_attrs else nn.Identity()

        self.img_zs_attr_weights = Hadamard(n_zs_attrs, bias=use_bias) if n_zs_attrs else nn.Identity()
        self.regions_zs_attr_weights = Hadamard(n_zs_attrs, bias=use_bias) if n_zs_attrs else nn.Identity()

        # Detect component concepts solely from the full image, not from regions
        self.set_num_component_concepts(n_component_concepts)

        self.feature_groups = nn.ModuleDict()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, features: ConceptPredictorFeatures) -> ConceptPredictorOutput:
        if features.all_scores is None: # If scores are not provided for feature_group-weighting, calculate them
            region_weights = features.region_weights.unsqueeze(-1) # (n_regions, 1)

            # Image and region feature scores
            if self.use_full_img:
                img_score = self.img_features_predictor(features.image_features) # (1, 1)
            else:
                img_score = torch.tensor([[]], device=region_weights.device) # (1, 0)

            if self.use_regions and self.use_region_features:
                region_scores = self.region_features_predictor(features.region_features) # (n_regions, 1)
                region_scores = region_scores * region_weights # (n_regions, 1)
                region_score = region_scores.sum(dim=0, keepdim=True) # (1, 1)
            else:
                region_scores = torch.tensor([[]], device=region_weights.device) # (1, 0)
                region_score = torch.tensor([[]], device=region_weights.device) # (1, 0)

            # Trained attributes
            if self.use_full_img:
                trained_attr_img_scores = features.trained_attr_img_scores # (1, n_trained_attrs); possibly (1, 0)
                trained_attr_img_score = trained_attr_img_scores # (1, n_trained_attrs); possibly (1, 0)
            else:
                trained_attr_img_score = torch.tensor([[]], device=region_weights.device) # (1, 0)

            if self.use_regions and features.trained_attr_region_scores.numel(): # Using regions and there are trained attributes
                trained_attr_region_scores = features.trained_attr_region_scores * region_weights # (n_regions, n_trained_attrs); possibly (n_regions, 0)
                trained_attr_region_score = trained_attr_region_scores.sum(dim=0, keepdim=True) # (1, n_trained_attrs)
            else:
                trained_attr_region_scores = torch.tensor([[]], device=region_weights.device) # (1, 0)
                trained_attr_region_score = torch.tensor([[]], device=region_weights.device) # (1, 0)

            # Zero shot attributes
            if self.use_full_img:
                zs_attr_img_scores = features.zs_attr_img_scores # (1, n_zs_attrs); possibly (1, 0)
                zs_attr_img_score = zs_attr_img_scores # (1, n_zs_attrs); possibly (1, 0)
            else:
                zs_attr_img_score = torch.tensor([[]], device=region_weights.device) # (1, 0)

            if self.use_regions and features.zs_attr_region_scores.numel(): # Using regions and there are zero-shot attributes
                zs_attr_region_scores = features.zs_attr_region_scores * region_weights # (n_regions, n_zs_attrs); possibly (n_regions, 0)
                zs_attr_region_score = zs_attr_region_scores.sum(dim=0, keepdim=True) # (1, n_zs_attrs)
            else:
                zs_attr_region_scores = torch.tensor([[]], device=region_weights.device) # (1, 0)
                zs_attr_region_score = torch.tensor([[]], device=region_weights.device) # (1, 0)

            # Component concept scores
            component_concept_scores = features.component_concept_scores # (1, n_component_concepts)

            # Concatenate all scores for layer norm
            all_scores = torch.cat([
                img_score, # (1, 1)
                region_score, # (1, 1)
                trained_attr_img_score, # (1, n_trained_attrs)
                trained_attr_region_score, # (1, n_trained_attrs)
                zs_attr_img_score, # (1, n_zs_attrs)
                zs_attr_region_score, # (1, n_zs_attrs)
                component_concept_scores # (1, n_component_concepts)
            ], dim=1) # (1, 1 + 1 + 2*n_trained_attrs + 2*n_zs_attrs)

            all_scores = self.ln(all_scores)
            all_scores = self.prob_scaler(all_scores)

            # Split scores and apply weights of linear model
            (
                img_score, region_score,
                trained_attr_img_score, trained_attr_region_score,
                zs_attr_img_score, zs_attr_region_score,
                component_concept_scores
            ) = all_scores.split((
                int(self.use_full_img), # 1 or 0
                int(self.use_regions and self.use_region_features), # 1 or 0
                self.use_full_img * self.n_trained_attrs, # n_trained_attrs or 0
                self.use_regions * self.n_trained_attrs, # n_trained_attrs or 0
                self.use_full_img * self.n_zs_attrs, # n_zs_attrs or 0
                self.use_regions * self.n_zs_attrs, # n_zs_attrs or 0
                self.n_component_concepts
            ), dim=1)

            # Regions and images were set to (X, 0) if not using them, so no need to check use_full_img or use_regions
            # Trained and zero-shot weights are Identity if there are zero of either, so no need to check len > 0
            img_score = self.img_features_weight(img_score)
            region_score = self.region_features_weight(region_score)
            trained_attr_img_score = self.img_trained_attr_weights(trained_attr_img_score)
            trained_attr_region_score = self.regions_trained_attr_weights(trained_attr_region_score)
            zs_attr_img_score = self.img_zs_attr_weights(zs_attr_img_score)
            zs_attr_region_score = self.regions_zs_attr_weights(zs_attr_region_score)
            component_concept_scores = self.component_concept_weights(component_concept_scores)

            all_scores = torch.cat([
                img_score, # (1, 1)
                region_score, # (1, 1)
                trained_attr_img_score, # (1, n_trained_attrs)
                trained_attr_region_score, # (1, n_trained_attrs)
                zs_attr_img_score, # (1, n_zs_attrs)
                zs_attr_region_score, # (1, n_zs_attrs)
                component_concept_scores # (1, n_component_concepts)
            ], dim=1)

        else:
            all_scores = features.all_scores

        for group in self.feature_groups:
            all_scores = group(all_scores)

        # Construct output
        if features.all_scores is None: # Intermediate results computed here
            output = ConceptPredictorOutput(
                img_score=img_score.detach().squeeze() if self.use_full_img else None,
                region_scores=region_scores.detach().squeeze() if self.use_regions else None,
                region_score=region_score.detach().squeeze() if self.use_regions else None,
                trained_attr_img_scores=trained_attr_img_scores.detach().squeeze() if self.use_full_img else None,
                trained_attr_region_scores=trained_attr_region_scores.detach() if self.use_regions else None,
                zs_attr_img_scores=zs_attr_img_scores.detach().squeeze() if self.use_full_img else None,
                zs_attr_region_scores=zs_attr_region_scores.detach() if self.use_regions else None,
                component_concept_scores=component_concept_scores.detach().squeeze(),
                all_scores_weighted=all_scores.detach().squeeze(),
                cum_score=all_scores.sum()
            )

        else: # Intermediate results computed externally;
            output = ConceptPredictorOutput(
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

    def set_num_component_concepts(self, n_component_concepts: int):
        self.n_component_concepts = n_component_concepts
        self.component_concept_weights = Hadamard(n_component_concepts, bias=self.use_bias) if n_component_concepts else nn.Identity()