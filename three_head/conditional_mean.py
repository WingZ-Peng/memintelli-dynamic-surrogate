from typing import Tuple

import torch
from torch import nn


class TileConditionalMean(nn.Module):
    """Predict the conditional residual mean from boundary-aligned tile features."""

    def __init__(
        self,
        feature_dim: int,
        target_dim: int,
        hidden_dim: int = 128,
        coordinate_dim: int = 16,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or target_dim < 1:
            raise ValueError("feature_dim and target_dim must be positive")
        self.feature_dim = feature_dim
        self.target_dim = target_dim
        self.coordinate_embedding = nn.Parameter(
            torch.empty(target_dim, coordinate_dim)
        )
        nn.init.normal_(self.coordinate_embedding, mean=0.0, std=0.02)
        self.local_encoder = nn.Sequential(
            nn.Linear(feature_dim + coordinate_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_scale", torch.ones(feature_dim))
        self.register_buffer("target_mean", torch.zeros(target_dim))
        self.register_buffer("target_scale", torch.ones(target_dim))
        self.register_buffer(
            "normalization_ready", torch.tensor(False, dtype=torch.bool)
        )

    @torch.no_grad()
    def configure_normalization(
        self, features: torch.Tensor, targets: torch.Tensor
    ) -> None:
        if features.ndim != 3 or features.shape[1:] != (
            self.target_dim,
            self.feature_dim,
        ):
            raise ValueError("features must have shape [C,D,F]")
        if targets.ndim != 2 or targets.shape[1] != self.target_dim:
            raise ValueError("targets must have shape [C,D]")
        flattened_features = features.reshape(-1, self.feature_dim)
        self.feature_mean.copy_(flattened_features.mean(dim=0))
        self.feature_scale.copy_(
            flattened_features.std(dim=0, unbiased=False).clamp_min(1e-6)
        )
        self.target_mean.copy_(targets.mean(dim=0))
        self.target_scale.copy_(
            targets.std(dim=0, unbiased=False).clamp_min(1e-6)
        )
        self.normalization_ready.fill_(True)

    def _require_normalization(self) -> None:
        if not bool(self.normalization_ready.item()):
            raise RuntimeError("configure_normalization() must be called first")

    def normalized_prediction(self, features: torch.Tensor) -> torch.Tensor:
        self._require_normalization()
        normalized = (features - self.feature_mean) / self.feature_scale
        coordinate = self.coordinate_embedding.unsqueeze(0).expand(
            normalized.shape[0], -1, -1
        )
        local = self.local_encoder(torch.cat((normalized, coordinate), dim=-1))
        global_mean = local.mean(dim=1)
        global_max = local.amax(dim=1)
        global_hidden = self.global_encoder(
            torch.cat((global_mean, global_max), dim=-1)
        )
        global_expanded = global_hidden.unsqueeze(1).expand_as(local)
        return self.fusion(
            torch.cat((local, global_expanded), dim=-1)
        ).squeeze(-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.normalized_prediction(features)
        return self.target_mean + normalized * self.target_scale

    def normalized_targets(self, targets: torch.Tensor) -> torch.Tensor:
        self._require_normalization()
        return (targets - self.target_mean) / self.target_scale

    def prediction_and_normalized_target(
        self, features: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.normalized_prediction(features), self.normalized_targets(targets)
