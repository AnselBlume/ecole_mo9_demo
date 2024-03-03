import torch
import torch.nn as nn
from .features import ImageFeatures
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

    def to(self, device, detach=True):
        '''
            Detaches and shifts all tensors to the specified device
        '''
        for field in self.__dataclass_fields__:
            attr = getattr(self, field)

            if isinstance(attr, torch.Tensor):
                if detach:
                    attr = attr.detach()

                setattr(self, field, attr.to(device))

        return self

    def cpu(self):
        '''
            Moves all tensors to the CPU.
        '''
        return self.to('cpu')

    def cuda(self):
        '''
            Moves all tensors to the GPU.
        '''
        return self.to('cuda')

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
        use_bias=True,
        use_ln=True,
        use_full_img=True,
        use_regions=True
    ):
        super().__init__()

        self.img_feature_dim = img_feature_dim
        self.region_feature_dim = region_feature_dim
        self.n_trained_attrs = n_trained_attrs
        self.n_zs_attrs = n_zs_attrs
        self.use_bias = use_bias
        self.use_ln = use_ln
        self.use_full_img = use_full_img
        self.use_regions = use_regions

        # Modules
        self.n_features = (
            use_full_img + use_regions
            + (use_full_img + use_regions) * n_trained_attrs
            + (use_full_img + use_regions) * n_zs_attrs
        )

        self.ln = nn.LayerNorm(self.n_features) if use_ln else nn.Identity()

        self.img_features_predictor = nn.Linear(img_feature_dim, 1, bias=use_bias) # This will only be used if use_full_img is True
        self.img_features_weight = nn.Linear(1, 1, bias=use_bias) if self.use_full_img else nn.Identity()

        self.region_features_predictor = nn.Linear(region_feature_dim, 1, bias=use_bias) # This will only be used if use_regions is True
        self.region_features_weight = nn.Linear(1, 1, bias=use_bias) if self.use_regions else nn.Identity()

        self.img_trained_attr_weights = Hadamard(n_trained_attrs, bias=use_bias) if n_trained_attrs else nn.Identity()
        self.regions_trained_attr_weights = Hadamard(n_trained_attrs, bias=use_bias) if n_trained_attrs else nn.Identity()

        self.img_zs_attr_weights = Hadamard(n_zs_attrs, bias=use_bias) if n_zs_attrs else nn.Identity()
        self.regions_zs_attr_weights = Hadamard(n_zs_attrs, bias=use_bias) if n_zs_attrs else nn.Identity()

        self.feature_groups = nn.ModuleDict()

    def forward(self, img_feats: ImageFeatures) -> ConceptPredictorOutput:
        if img_feats.all_scores is None: # If scores are not provided for feature_group-weighting, calculate them
            region_weights = img_feats.region_weights.unsqueeze(-1) # (n_regions, 1)

            # Image and region feature scores
            if self.use_full_img:
                img_score = self.img_features_predictor(img_feats.image_features) # (1, 1)
            else:
                img_score = torch.tensor([[]], device=region_weights.device) # (1, 0)

            if self.use_regions:
                region_scores = self.region_features_predictor(img_feats.region_features) # (n_regions, 1)
                region_scores = region_scores * region_weights # (n_regions, 1)
                region_score = region_scores.sum(dim=0, keepdim=True) # (1, 1)
            else:
                region_score = torch.tensor([[]], device=region_weights.device) # (1, 0)

            # Trained attributes
            if self.use_full_img:
                trained_attr_img_scores = img_feats.trained_attr_img_scores # (1, n_trained_attrs); possibly (1, 0)
                trained_attr_img_score = trained_attr_img_scores # (1, n_trained_attrs); possibly (1, 0)
            else:
                trained_attr_img_score = torch.tensor([[]], device=region_weights.device) # (1, 0)

            if self.use_regions:
                trained_attr_region_scores = img_feats.trained_attr_region_scores * region_weights # (n_regions, n_trained_attrs); possibly (n_regions, 0)
                trained_attr_region_score = trained_attr_region_scores.sum(dim=0, keepdim=True) # (1, n_trained_attrs)
            else:
                trained_attr_region_score = torch.tensor([[]], device=region_weights.device) # (1, 0)

            # Zero shot attributes
            if self.use_full_img:
                zs_attr_img_scores = img_feats.zs_attr_img_scores # (1, n_zs_attrs); possibly (1, 0)
                zs_attr_img_score = zs_attr_img_scores # (1, n_zs_attrs); possibly (1, 0)
            else:
                zs_attr_img_score = torch.tensor([[]], device=region_weights.device) # (1, 0)

            if self.use_regions:
                zs_attr_region_scores = img_feats.zs_attr_region_scores * region_weights # (n_regions, n_zs_attrs); possibly (n_regions, 0)
                zs_attr_region_score = zs_attr_region_scores.sum(dim=0, keepdim=True) # (1, n_zs_attrs)
            else:
                zs_attr_region_score = torch.tensor([[]], device=region_weights.device) # (1, 0)

            # Concatenate all scores for layer norm
            all_scores = torch.cat([
                img_score, # (1, 1)
                region_score, # (1, 1)
                trained_attr_img_score, # (1, n_trained_attrs)
                trained_attr_region_score, # (1, n_trained_attrs)
                zs_attr_img_score, # (1, n_zs_attrs)
                zs_attr_region_score # (1, n_zs_attrs)
            ], dim=1) # (1, 1 + 1 + 2*n_trained_attrs + 2*n_zs_attrs)

            all_scores = self.ln(all_scores)

            # Split scores and apply weights of linear model
            img_score, region_score, trained_attr_img_score, trained_attr_region_score, zs_attr_img_score, zs_attr_region_score = all_scores.split((
                int(self.use_full_img), # 1 or 0
                int(self.use_regions), # 1 or 0
                self.use_full_img * self.n_trained_attrs, # n_trained_attrs or 0
                self.use_regions * self.n_trained_attrs, # n_trained_attrs or 0
                self.use_full_img * self.n_zs_attrs, # n_zs_attrs or 0
                self.use_regions * self.n_zs_attrs # n_zs_attrs or 0
            ), dim=1)

            # Regions and images were set to (X, 0) if not using them, so no need to check use_full_img or use_regions
            # Trained and zero-shot weights are Identity if there are zero of either, so no need to check len > 0
            img_score = self.img_features_weight(img_score)
            region_score = self.region_features_weight(region_score)
            trained_attr_img_score = self.img_trained_attr_weights(trained_attr_img_score)
            trained_attr_region_score = self.regions_trained_attr_weights(trained_attr_region_score)
            zs_attr_img_score = self.img_zs_attr_weights(zs_attr_img_score)
            zs_attr_region_score = self.regions_zs_attr_weights(zs_attr_region_score)

            all_scores = torch.cat([
                img_score, # (1, 1)
                region_score, # (1, 1)
                trained_attr_img_score, # (1, n_trained_attrs)
                trained_attr_region_score, # (1, n_trained_attrs)
                zs_attr_img_score, # (1, n_zs_attrs)
                zs_attr_region_score # (1, n_zs_attrs)
            ], dim=1)

        else:
            all_scores = img_feats.all_scores

        for group in self.feature_groups:
            all_scores = group(all_scores)

        # Construct output
        if img_feats.all_scores is None: # Intermediate results computed here
            output = ConceptPredictorOutput(
                img_score=img_score.detach().squeeze() if self.use_full_img else None,
                region_scores=region_scores.detach().squeeze() if self.use_regions else None,
                region_score=region_score.detach().squeeze() if self.use_regions else None,
                trained_attr_img_scores=trained_attr_img_scores.detach().squeeze() if self.use_full_img else None,
                trained_attr_region_scores=trained_attr_region_scores.detach() if self.use_regions else None,
                zs_attr_img_scores=zs_attr_img_scores.detach().squeeze() if self.use_full_img else None,
                zs_attr_region_scores=zs_attr_region_scores.detach() if self.use_regions else None,
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

