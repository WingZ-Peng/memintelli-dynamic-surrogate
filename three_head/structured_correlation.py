from __future__ import annotations

import torch
from torch import nn

from .structure import CoordinateStructure


EPS = 1e-12


def grouped_projection(
    loading: torch.Tensor,
    permutation: torch.Tensor,
    inverse: torch.Tensor,
    group_count: int,
    group_size: int,
    sample_count: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Project independent per-group noise onto coordinates that share it.

    Coordinates are permuted so each source group occupies a contiguous block,
    which turns the sharing pattern into a plain reshape. Drawing noise per group
    instead of per coordinate keeps the tensor at [C,N,groups,source_dim] rather
    than [C,N,D,source_dim].
    """
    context_count, target_dim, source_dim = loading.shape
    sorted_loading = loading[:, permutation].reshape(
        context_count, group_count, group_size, source_dim
    )
    noise = torch.randn(
        context_count,
        sample_count,
        group_count,
        source_dim,
        dtype=loading.dtype,
        device=loading.device,
        generator=generator,
    )
    projected = torch.einsum("cgif,cngf->cngi", sorted_loading, noise)
    return projected.reshape(context_count, sample_count, target_dim)[:, :, inverse]


def structured_sample(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    residual: torch.Tensor,
    structure: CoordinateStructure,
    variances: torch.Tensor,
    sample_count: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Centered samples from the source factor model, scaled to `variances`.

    Shared by the trained head and by the parameter-free analytic predictor, so
    both are sampled through exactly the same generative path.
    """
    voltage = grouped_projection(
        alpha,
        structure.voltage_permutation,
        structure.voltage_inverse,
        structure.voltage_group_count,
        structure.voltage_group_size,
        sample_count,
        generator,
    )
    conductance = grouped_projection(
        beta,
        structure.conductance_permutation,
        structure.conductance_inverse,
        structure.conductance_group_count,
        structure.conductance_group_size,
        sample_count,
        generator,
    )
    independent = torch.randn(
        alpha.shape[0],
        sample_count,
        alpha.shape[1],
        dtype=alpha.dtype,
        device=alpha.device,
        generator=generator,
    )
    standardized = (
        voltage + conductance + residual.clamp_min(0.0).sqrt().unsqueeze(1) * independent
    )
    return standardized * variances.sqrt().unsqueeze(1)


# --- Previous parameterization ------------------------------------------------
# src/structured_covariance.py :: AnchoredTileConditionalCorrelationGaussian
#
#     raw_factor = self.shared_raw_factor.unsqueeze(0) + delta
#     denominator = 1.0 + raw_factor.square().sum(dim=-1)
#     diagonal = denominator.reciprocal()
#     factor = raw_factor / denominator.sqrt().unsqueeze(-1)
#     R = diag(diagonal) + factor @ factor.T          # factor is [C, D, rank=64]
#
# `shared_raw_factor` is initialized from a rank-64 truncation of the *context
# averaged* empirical correlation, so every context starts at the population mean
# and the network must transport it back. Two structural problems:
#
#   1. A dense [D, 64] factor puts mass on all 9,900 off-diagonal entries, but
#      only ~1,300 of them can physically be non-zero. It cannot express an exact
#      zero pattern cheaply.
#   2. The true covariance is a sum of two block-diagonal terms under two
#      different permutations (blocks by (k_block, row) and by (k_block, out)).
#      Its combined rank reaches D=100, so a global rank-64 factor is a lossy
#      approximation of a full-rank object no matter how long it trains.
# -----------------------------------------------------------------------------


class StructuredSourceCorrelation(nn.Module):
    """Correlation as a factor model over the *physical* noise sources.

    Each coordinate d = (k_block, row, out) is written as a unit-variance sum

        S_d = <alpha_d, eps_voltage[k, row]> + <beta_d, eps_conductance[k, out]>
              + sqrt(1 - |alpha_d|^2 - |beta_d|^2) * eta_d

    with independent standard normal eps/eta. Every structural property then holds
    by construction rather than by fitting:

      * coordinates sharing no source are exactly uncorrelated;
      * the matrix is positive semi-definite for any parameter value;
      * the diagonal is exactly one;
      * sampling is native and needs no Cholesky factorization.

    The network predicts only a bounded correction to the analytic loading that
    `features.py` already computes, and the correction is zero-initialized, so
    training *starts* at the closed-form first-order solution.
    """

    def __init__(
        self,
        feature_dim: int,
        structure: CoordinateStructure,
        hidden_dim: int = 128,
        coordinate_dim: int = 16,
        source_dim: int = 24,
        residual_logit_bias: float = 6.0,
        delta_bound: float = 1.0,
    ) -> None:
        super().__init__()
        if feature_dim < 2 * source_dim:
            raise ValueError("feature_dim must contain the 2*source_dim anchor")
        self.feature_dim = feature_dim
        self.target_dim = structure.dim
        self.source_dim = source_dim
        self.residual_logit_bias = residual_logit_bias
        self.delta_bound = delta_bound
        self.voltage_group_count = structure.voltage_group_count
        self.conductance_group_count = structure.conductance_group_count
        self.voltage_group_size = structure.voltage_group_size
        self.conductance_group_size = structure.conductance_group_size

        self.coordinate_embedding = nn.Parameter(
            torch.zeros(structure.dim, coordinate_dim)
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
        )
        self.delta_head = nn.Linear(hidden_dim, 2 * source_dim)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self.residual_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_scale", torch.ones(feature_dim))
        self.register_buffer(
            "normalization_ready", torch.tensor(False, dtype=torch.bool)
        )
        self.register_buffer("voltage_mask", structure.voltage_mask.clone())
        self.register_buffer("conductance_mask", structure.conductance_mask.clone())
        self.register_buffer(
            "voltage_permutation", structure.voltage_permutation.clone()
        )
        self.register_buffer("voltage_inverse", structure.voltage_inverse.clone())
        self.register_buffer(
            "conductance_permutation", structure.conductance_permutation.clone()
        )
        self.register_buffer(
            "conductance_inverse", structure.conductance_inverse.clone()
        )

    @torch.no_grad()
    def configure_normalization(self, features: torch.Tensor) -> None:
        if features.ndim != 3 or features.shape[1:] != (
            self.target_dim,
            self.feature_dim,
        ):
            raise ValueError("features must have shape [C,D,F]")
        flattened = features.reshape(-1, self.feature_dim)
        self.feature_mean.copy_(flattened.mean(dim=0))
        self.feature_scale.copy_(flattened.std(dim=0, unbiased=False).clamp_min(1e-6))
        self.normalization_ready.fill_(True)

    def _require_normalization(self) -> None:
        if not bool(self.normalization_ready.item()):
            raise RuntimeError("configure_normalization() must be called first")

    def _hidden(self, features: torch.Tensor) -> torch.Tensor:
        self._require_normalization()
        normalized = (features - self.feature_mean) / self.feature_scale
        coordinates = self.coordinate_embedding.unsqueeze(0).expand(
            features.shape[0], -1, -1
        )
        local = self.local_encoder(torch.cat((normalized, coordinates), dim=-1))
        pooled = torch.cat((local.mean(dim=1), local.amax(dim=1)), dim=-1)
        global_hidden = self.global_encoder(pooled)
        return self.fusion(
            torch.cat((local, global_hidden.unsqueeze(1).expand_as(local)), dim=-1)
        )

    def source_parameters(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (alpha, beta, explained_fraction, anchor_deviation).

        The anchor is read from the *raw* feature vector, not the normalized one:
        `features[..., :48]` is already the unit-norm analytic source loading.
        `anchor_deviation` is returned so the loss can shrink the learned
        correction back toward the closed-form solution.
        """
        hidden = self._hidden(features)
        anchor = features[..., : 2 * self.source_dim]
        delta = self.delta_bound * torch.tanh(
            self.delta_head(hidden) / self.delta_bound
        )
        direction = anchor + delta
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        explained = torch.sigmoid(
            self.residual_logit_bias + self.residual_head(hidden).squeeze(-1)
        )
        loading = direction * explained.sqrt().unsqueeze(-1)
        return (
            loading[..., : self.source_dim],
            loading[..., self.source_dim :],
            explained,
            delta,
        )

    def correlation_matrix_and_deviation(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha, beta, explained, delta = self.source_parameters(features)
        voltage = torch.einsum("cdf,cef->cde", alpha, alpha) * self.voltage_mask
        conductance = (
            torch.einsum("cdf,cef->cde", beta, beta) * self.conductance_mask
        )
        correlation = voltage + conductance
        diagonal = torch.diagonal(correlation, dim1=-2, dim2=-1)
        correlation = (
            correlation
            - torch.diag_embed(diagonal)
            + torch.diag_embed(torch.ones_like(explained))
        )
        return correlation, delta

    def correlation_matrix(self, features: torch.Tensor) -> torch.Tensor:
        return self.correlation_matrix_and_deviation(features)[0]

    @torch.no_grad()
    def sample(
        self,
        features: torch.Tensor,
        variances: torch.Tensor,
        sample_count: int,
        structure: CoordinateStructure,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Centered samples scaled to `variances`, matching the base head's contract.

        `structure` is passed in rather than held on the module so the caller
        controls its device; the module's own mask buffers stay authoritative for
        `correlation_matrix`.
        """
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        if variances.shape != features.shape[:2]:
            raise ValueError("variances must have shape [C,D]")
        if torch.any(variances <= 0):
            raise ValueError("variances must be positive")
        alpha, beta, explained, _ = self.source_parameters(features)
        return structured_sample(
            alpha,
            beta,
            1.0 - explained,
            structure,
            variances,
            sample_count,
            generator,
        )
