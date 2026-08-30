from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


EPS = 1e-12

# Feature layout of the base `deterministic_shared_source_loading` contract
# (ideal_dynamic/features.py, 118 dimensions):
#   [  0: 48]  normalized source loading = [voltage(24) ; conductance(24)]
#   [ 48: 72]  flattened quantized voltage
#   [ 72: 96]  normalized conductance level index
#   [ 96:105]  ADC transition activity (3x3 slice pairs)
#   [105:114]  ADC threshold distance (3x3 slice pairs)
#   [    114]  source energy
#   [115:118]  coordinate identity
ANCHOR_SLICE = (0, 48)
SOURCE_ENERGY_INDEX = 114


@dataclass(frozen=True)
class CoordinateStructure:
    """Which S_tile coordinates physically share a noise source.

    A coordinate is (k_block, row_index, out_index). It reads voltages on its own
    row within its own k_block, and conductance cells in its own output column
    within its own k_block. Two coordinates are therefore functions of *disjoint*
    random variables unless they share a k_block AND (a row OR an output column);
    their covariance is then exactly zero, not merely small.
    """

    dim: int
    k_block: torch.Tensor
    row_index: torch.Tensor
    out_index: torch.Tensor
    voltage_mask: torch.Tensor
    conductance_mask: torch.Tensor
    support_mask: torch.Tensor
    voltage_group: torch.Tensor
    conductance_group: torch.Tensor
    voltage_group_count: int
    conductance_group_count: int
    voltage_permutation: torch.Tensor
    conductance_permutation: torch.Tensor
    voltage_inverse: torch.Tensor
    conductance_inverse: torch.Tensor
    voltage_group_size: int
    conductance_group_size: int

    def to(self, device: torch.device) -> "CoordinateStructure":
        moved = {
            field: getattr(self, field).to(device)
            if isinstance(getattr(self, field), torch.Tensor)
            else getattr(self, field)
            for field in self.__dataclass_fields__
        }
        return CoordinateStructure(**moved)


def build_coordinate_structure(
    valid_mask: torch.Tensor, configuration: Mapping[str, Any]
) -> CoordinateStructure:
    coordinates = torch.nonzero(valid_mask, as_tuple=False)
    tile_rows, tile_cols = valid_mask.shape[-2], valid_mask.shape[-1]
    k_block = coordinates[:, 1].contiguous()
    row_index = (coordinates[:, 0] * tile_rows + coordinates[:, 3]).contiguous()
    out_index = (coordinates[:, 2] * tile_cols + coordinates[:, 4]).contiguous()
    dim = coordinates.shape[0]

    same_k = k_block[:, None] == k_block[None, :]
    same_row = row_index[:, None] == row_index[None, :]
    same_out = out_index[:, None] == out_index[None, :]
    eye = torch.eye(dim, dtype=torch.bool)

    voltage_mask = same_k & same_row
    conductance_mask = same_k & same_out
    support_mask = (voltage_mask | conductance_mask) & ~eye

    rows = int(row_index.max().item()) + 1
    outs = int(out_index.max().item()) + 1
    voltage_group = k_block * rows + row_index
    conductance_group = k_block * outs + out_index
    voltage_group = torch.unique(voltage_group, return_inverse=True)[1]
    conductance_group = torch.unique(conductance_group, return_inverse=True)[1]
    voltage_group_count = int(voltage_group.max().item()) + 1
    conductance_group_count = int(conductance_group.max().item()) + 1

    voltage_permutation = torch.argsort(
        voltage_group * dim + torch.arange(dim), stable=True
    )
    conductance_permutation = torch.argsort(
        conductance_group * dim + torch.arange(dim), stable=True
    )
    voltage_inverse = torch.argsort(voltage_permutation)
    conductance_inverse = torch.argsort(conductance_permutation)

    voltage_sizes = torch.bincount(voltage_group)
    conductance_sizes = torch.bincount(conductance_group)
    if int(voltage_sizes.min()) != int(voltage_sizes.max()):
        raise ValueError("Voltage groups are not uniform; regular reshape is invalid")
    if int(conductance_sizes.min()) != int(conductance_sizes.max()):
        raise ValueError("Conductance groups are not uniform; regular reshape is invalid")

    if int(configuration["input_shape"][0]) != rows:
        raise ValueError("Derived row count differs from the fixed input shape")
    if int(configuration["weight_shape"][1]) != outs:
        raise ValueError("Derived output count differs from the fixed weight shape")

    return CoordinateStructure(
        dim=dim,
        k_block=k_block,
        row_index=row_index,
        out_index=out_index,
        voltage_mask=voltage_mask,
        conductance_mask=conductance_mask,
        support_mask=support_mask,
        voltage_group=voltage_group,
        conductance_group=conductance_group,
        voltage_group_count=voltage_group_count,
        conductance_group_count=conductance_group_count,
        voltage_permutation=voltage_permutation,
        conductance_permutation=conductance_permutation,
        voltage_inverse=voltage_inverse,
        conductance_inverse=conductance_inverse,
        voltage_group_size=int(voltage_sizes[0]),
        conductance_group_size=int(conductance_sizes[0]),
    )


def analytic_anchor(correlation_features: torch.Tensor) -> torch.Tensor:
    """The unit-norm [voltage(24) ; conductance(24)] source loading, per coordinate.

    features.py already stores `source_loading / ||source_loading||`, so the anchor
    needs no rescaling: its squared norm is exactly one, which makes the analytic
    covariance a correlation matrix by construction.
    """
    start, stop = ANCHOR_SLICE
    return correlation_features[..., start:stop].float()


def analytic_correlation(
    anchor: torch.Tensor, structure: CoordinateStructure
) -> torch.Tensor:
    """Closed-form first-order correlation. No learned parameters."""
    source_dim = anchor.shape[-1] // 2
    alpha = anchor[..., :source_dim]
    beta = anchor[..., source_dim:]
    voltage = torch.einsum("cdf,cef->cde", alpha, alpha)
    conductance = torch.einsum("cdf,cef->cde", beta, beta)
    correlation = (
        voltage * structure.voltage_mask + conductance * structure.conductance_mask
    )
    diagonal = torch.diagonal(correlation, dim1=-2, dim2=-1)
    identity = torch.eye(
        structure.dim, dtype=correlation.dtype, device=correlation.device
    )
    return correlation - torch.diag_embed(diagonal) + identity.unsqueeze(0)


def empirical_correlation(samples: torch.Tensor) -> torch.Tensor:
    """Unchanged from ideal_dynamic/training.py so the labels stay comparable."""
    centered = samples - samples.mean(dim=1, keepdim=True)
    covariance = torch.einsum("cnd,cne->cde", centered, centered) / max(
        samples.shape[1] - 1, 1
    )
    variance = torch.diagonal(covariance, dim1=-2, dim2=-1).clamp_min(EPS)
    inverse = variance.rsqrt()
    correlation = covariance * inverse.unsqueeze(-1) * inverse.unsqueeze(-2)
    diagonal = torch.diagonal(correlation, dim1=-2, dim2=-1)
    identity = torch.eye(
        correlation.shape[-1], dtype=correlation.dtype, device=correlation.device
    )
    return correlation - torch.diag_embed(diagonal) + identity.unsqueeze(0)
