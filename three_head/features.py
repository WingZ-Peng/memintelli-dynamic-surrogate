from __future__ import annotations

import math
from typing import Any, Mapping

import torch


EPS = 1e-12


def _read_sigma_for_cells(
    conductance: torch.Tensor,
    levels: torch.Tensor,
    read_variation: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    level_index = (
        conductance.unsqueeze(-1) - levels.reshape(1, -1)
    ).abs().argmin(dim=-1)
    if isinstance(read_variation, Mapping):
        values = torch.tensor(
            [
                float(
                    read_variation.get(
                        index, read_variation.get(str(index), 0.0)
                    )
                )
                for index in range(levels.numel())
            ],
            dtype=conductance.dtype,
            device=conductance.device,
        )
        sigma = values[level_index]
    else:
        sigma = torch.full_like(conductance, float(read_variation))
    normalized_level = level_index.to(conductance.dtype) / max(
        levels.numel() - 1, 1
    )
    return sigma, normalized_level


def _adc_transition_gain(
    mean_code: torch.Tensor, code_standard_deviation: torch.Tensor
) -> torch.Tensor:
    standard_deviation = code_standard_deviation.clamp_min(1e-5)
    center = torch.round(mean_code).unsqueeze(-1)
    offsets = torch.arange(
        -4, 5, dtype=mean_code.dtype, device=mean_code.device
    )
    thresholds = center + offsets + 0.5
    normalized = (
        thresholds - mean_code.unsqueeze(-1)
    ) / standard_deviation.unsqueeze(-1)
    density = torch.exp(-0.5 * normalized.square()) / (
        math.sqrt(2.0 * math.pi) * standard_deviation.unsqueeze(-1)
    )
    return density.sum(dim=-1).clamp(0.0, 2.0)


def build_head_features(data: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    required = {
        "inputs",
        "ideal_g",
        "valid_mask",
        "nominal_s_tile",
        "nominal_v_quant",
        "nominal_pre_adc",
        "nominal_post_adc",
        "conductance_levels",
        "phase_coefficients",
        "configuration",
    }
    missing = required.difference(data)
    if missing:
        raise KeyError(f"Feature payload is missing: {sorted(missing)}")

    inputs = data["inputs"].float()
    ideal_g = data["ideal_g"].float()
    valid_mask = data["valid_mask"]
    nominal = data["nominal_s_tile"].float()
    v_quant = data["nominal_v_quant"].float()
    pre_adc = data["nominal_pre_adc"].float()
    post_adc = data["nominal_post_adc"].float()
    levels = data["conductance_levels"].float()
    phase_coefficients = data["phase_coefficients"].float()
    config = data["configuration"]
    engine = config["engine"]
    coordinates = torch.nonzero(valid_mask, as_tuple=False)
    context_count = inputs.shape[0]
    tile_rows = valid_mask.shape[-2]
    tile_cols = valid_mask.shape[-1]
    lgs = float(engine["LGS"])
    conductance_span = float(engine["HGS"] - engine["LGS"])
    voltage_noise = float(engine["vnoise"])
    adc_ref = conductance_span * float(engine["vread"]) * v_quant.shape[-1]
    radc = float(engine["radc"])

    mean_features: list[torch.Tensor] = []
    variance_features: list[torch.Tensor] = []
    correlation_features: list[torch.Tensor] = []
    for coordinate_index, coordinate in enumerate(coordinates):
        row_block, k_block, out_block, tile_row, tile_col = (
            int(value.item()) for value in coordinate
        )
        row_index = row_block * tile_rows + tile_row
        out_index = out_block * tile_cols + tile_col
        local_v = v_quant[
            :, row_block, k_block, :, tile_row, :
        ].reshape(context_count, 3, 8)
        local_g = ideal_g[
            k_block, out_block, :, :, tile_col
        ].reshape(3, 8)
        read_sigma, level_index = _read_sigma_for_cells(
            local_g.reshape(-1), levels, engine["read_variation"]
        )
        read_sigma = read_sigma.reshape(3, 8)
        level_index = level_index.reshape(3, 8)
        local_pre = pre_adc[
            :, row_block, k_block, out_block, :, :, tile_row, tile_col
        ].reshape(context_count, 3, 3)
        local_post = post_adc[
            :, row_block, k_block, out_block, :, :, tile_row, tile_col
        ].reshape(context_count, 3, 3)

        expected_h = local_g * torch.exp(0.5 * read_sigma.square()) - lgs
        expected_h_squared = (
            local_g.square() * torch.exp(2.0 * read_sigma.square())
            - 2.0
            * lgs
            * local_g
            * torch.exp(0.5 * read_sigma.square())
            + lgs**2
        )
        expected_v_squared = local_v.square() * (1.0 + voltage_noise**2)
        expected_pre = torch.einsum("cik,jk->cij", local_v, expected_h)
        pre_second_sum = torch.einsum(
            "cik,jk->cij", expected_v_squared, expected_h_squared
        )
        term_mean_squared_sum = torch.einsum(
            "cik,jk->cij", local_v.square(), expected_h.square()
        )
        pre_variance = (pre_second_sum - term_mean_squared_sum).clamp_min(EPS)

        ideal_code = local_pre / adc_ref * (radc - 1.0)
        mean_code = expected_pre / adc_ref * (radc - 1.0)
        code_variance = pre_variance * ((radc - 1.0) / adc_ref) ** 2
        code_standard_deviation = code_variance.sqrt()
        signed_phase = ideal_code - torch.round(ideal_code)
        threshold_distance = 0.5 - signed_phase.abs()
        adc_activity = _adc_transition_gain(mean_code, code_standard_deviation)
        physical_gain = adc_activity / adc_ref

        voltage_loading = voltage_noise * local_v * torch.einsum(
            "ij,cij,jk->cik",
            phase_coefficients,
            physical_gain,
            expected_h,
        )
        conductance_loading = read_sigma.unsqueeze(0) * local_g.unsqueeze(0) * torch.einsum(
            "ij,cij,cik->cjk",
            phase_coefficients,
            physical_gain,
            local_v,
        )
        source_loading = torch.cat(
            (voltage_loading.reshape(context_count, -1),
             conductance_loading.reshape(context_count, -1)),
            dim=1,
        )
        source_energy = source_loading.square().sum(dim=1, keepdim=True)
        normalized_loading = source_loading / source_energy.sqrt().clamp_min(EPS)

        repeated_g = ((local_g - lgs) / conductance_span).reshape(1, -1).expand(
            context_count, -1
        )
        repeated_level = level_index.reshape(1, -1).expand(context_count, -1)
        repeated_read_sigma = read_sigma.reshape(1, -1).expand(
            context_count, -1
        )
        flattened_v = local_v.reshape(context_count, -1) / float(engine["vread"])
        nominal_value = nominal[:, coordinate_index : coordinate_index + 1]
        coordinates_feature = torch.tensor(
            [
                k_block / max(valid_mask.shape[1] - 1, 1),
                row_index / max(config["input_shape"][0] - 1, 1),
                out_index / max(config["weight_shape"][1] - 1, 1),
            ],
            dtype=inputs.dtype,
        ).reshape(1, 3).expand(context_count, -1)

        mean_scalars = torch.stack(
            (
                flattened_v.square().sum(dim=1),
                repeated_g.square().sum(dim=1),
                ideal_code.square().sum(dim=(1, 2)),
                threshold_distance.amin(dim=(1, 2)),
                threshold_distance.mean(dim=(1, 2)),
                code_standard_deviation.mean(dim=(1, 2)),
            ),
            dim=1,
        )
        mean_features.append(
            torch.cat(
                (
                    flattened_v,
                    repeated_g,
                    repeated_level,
                    ideal_code.reshape(context_count, -1),
                    local_post.reshape(context_count, -1),
                    signed_phase.reshape(context_count, -1),
                    threshold_distance.reshape(context_count, -1),
                    (mean_code - ideal_code).reshape(context_count, -1),
                    code_standard_deviation.reshape(context_count, -1),
                    nominal_value,
                    nominal_value.abs(),
                    nominal_value.square(),
                    mean_scalars,
                    coordinates_feature,
                ),
                dim=1,
            )
        )

        variance_scalars = torch.cat(
            (
                code_standard_deviation.mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                code_standard_deviation.amax(dim=(1, 2), keepdim=False).unsqueeze(1),
                adc_activity.mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                source_energy,
                threshold_distance.amin(dim=(1, 2), keepdim=False).unsqueeze(1),
                threshold_distance.mean(dim=(1, 2), keepdim=False).unsqueeze(1),
            ),
            dim=1,
        )
        variance_features.append(
            torch.cat(
                (
                    flattened_v.square(),
                    repeated_g.square(),
                    repeated_read_sigma,
                    ideal_code.reshape(context_count, -1),
                    code_standard_deviation.reshape(context_count, -1),
                    adc_activity.reshape(context_count, -1),
                    threshold_distance.reshape(context_count, -1),
                    code_variance.reshape(context_count, -1),
                    (mean_code - ideal_code).reshape(context_count, -1),
                    source_energy,
                    nominal_value.abs(),
                    nominal_value.square(),
                    variance_scalars,
                    coordinates_feature,
                ),
                dim=1,
            )
        )

        correlation_features.append(
            torch.cat(
                (
                    normalized_loading,
                    flattened_v,
                    repeated_level,
                    adc_activity.reshape(context_count, -1),
                    threshold_distance.reshape(context_count, -1),
                    source_energy,
                    coordinates_feature,
                ),
                dim=1,
            )
        )

    return {
        "mean_features": torch.stack(mean_features, dim=1),
        "variance_features": torch.stack(variance_features, dim=1),
        "correlation_features": torch.stack(correlation_features, dim=1),
    }
