from __future__ import annotations

import torch

from .structure import CoordinateStructure


EPS = 1e-12


def empirical_variance(samples: torch.Tensor) -> torch.Tensor:
    return samples.var(dim=1, unbiased=True).clamp_min(EPS)


def mean_nrmse_per_context(
    prediction: torch.Tensor, reference_samples: torch.Tensor
) -> torch.Tensor:
    scale = (
        reference_samples.var(dim=1, unbiased=True).mean(dim=-1).sqrt().clamp_min(EPS)
    )
    reference_mean = reference_samples.mean(dim=1)
    return (prediction - reference_mean).square().mean(dim=-1).sqrt() / scale


def variance_l1_per_context(
    prediction: torch.Tensor, reference_samples: torch.Tensor
) -> torch.Tensor:
    reference = empirical_variance(reference_samples)
    return (
        (prediction - reference).abs().mean(dim=-1)
        / reference.mean(dim=-1).clamp_min(EPS)
    )


def correlation_frobenius_per_context(
    prediction: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Legacy protocol metric. Its denominator includes the unit diagonal, which
    compresses every score; kept so numbers stay comparable to the v1 report."""
    return torch.linalg.matrix_norm(prediction - reference) / torch.linalg.matrix_norm(
        reference
    ).clamp_min(EPS)


def correlation_frobenius_offdiagonal_per_context(
    prediction: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Off-diagonal only, so identity scores exactly 1.0."""
    dim = prediction.shape[-1]
    off = ~torch.eye(dim, dtype=torch.bool, device=prediction.device)
    return torch.linalg.matrix_norm(
        (prediction - reference) * off
    ) / torch.linalg.matrix_norm(reference * off).clamp_min(EPS)


def correlation_frobenius_support_per_context(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    structure: CoordinateStructure,
) -> torch.Tensor:
    """Restricted to entries that can physically be non-zero. Everything outside
    the support is exactly zero in truth, so scoring there measures label noise."""
    support = structure.support_mask.to(prediction.device)
    return torch.linalg.matrix_norm(
        (prediction - reference) * support
    ) / torch.linalg.matrix_norm(reference * support).clamp_min(EPS)


def support_relative_squared_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    structure: CoordinateStructure,
) -> torch.Tensor:
    support = structure.support_mask.to(prediction.device)
    numerator = ((prediction - target) * support).square().sum(dim=(-2, -1))
    denominator = (target * support).square().sum(dim=(-2, -1)).clamp_min(EPS)
    return (numerator / denominator).mean()


def gaussian_negative_log_likelihood(
    correlation: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Unbiased in the label noise: the noise in `target` is zero-mean and enters
    only linearly, through the trace term. Normalized per coordinate."""
    dim = correlation.shape[-1]
    jitter = 1e-6 * torch.eye(dim, dtype=correlation.dtype, device=correlation.device)
    cholesky = torch.linalg.cholesky(correlation + jitter)
    log_determinant = 2.0 * torch.log(
        torch.diagonal(cholesky, dim1=-2, dim2=-1).clamp_min(EPS)
    ).sum(dim=-1)
    solved = torch.cholesky_solve(target, cholesky)
    trace = torch.diagonal(solved, dim1=-2, dim2=-1).sum(dim=-1)
    return (0.5 * (log_determinant + trace)).mean() / dim
