from typing import Any, Sequence

import torch


def build_valid_mask(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    s_tile_shape: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    row_blocks, k_blocks, out_blocks, tile_rows, tile_cols = s_tile_shape
    mask = torch.zeros(tuple(s_tile_shape), dtype=torch.bool, device=device)
    for row_block in range(row_blocks):
        for out_block in range(out_blocks):
            for tile_row in range(tile_rows):
                for tile_col in range(tile_cols):
                    row_index = row_block * tile_rows + tile_row
                    out_index = out_block * tile_cols + tile_col
                    if row_index < input_shape[0] and out_index < weight_shape[1]:
                        mask[row_block, :, out_block, tile_row, tile_col] = True
    if int(mask.sum().item()) == 0 or k_blocks < 1:
        raise ValueError("Derived S_tile mask is empty")
    return mask


def tail_scale(input_sliced: Any, weight_sliced: Any) -> torch.Tensor:
    if input_sliced.bw_e is None:
        out_block_max = torch.einsum(
            "nmij, mpij->nmpij",
            input_sliced.max_data,
            weight_sliced.max_data,
        )
        return (
            out_block_max
            / (2 ** (sum(input_sliced.slice_method) - 1) - 1)
            / (2 ** (sum(weight_sliced.slice_method) - 1) - 1)
        )
    out_block_e_bias = torch.einsum(
        "nmij, mpij->nmpij",
        2.0**input_sliced.e_bias,
        2.0**weight_sliced.e_bias,
    )
    return out_block_e_bias * 2.0 ** (
        4 - sum(input_sliced.slice_method) - sum(weight_sliced.slice_method)
    )


def output_from_valid_s_tile(
    valid_samples: torch.Tensor,
    valid_mask: torch.Tensor,
    s_tile_shape: Sequence[int],
    scale: torch.Tensor,
    rows: int,
    out_features: int,
) -> torch.Tensor:
    if valid_samples.ndim != 2:
        raise ValueError("valid_samples must have shape [N,D]")
    sample_count = valid_samples.shape[0]
    full = torch.zeros(
        (sample_count, *s_tile_shape),
        dtype=valid_samples.dtype,
        device=valid_samples.device,
    )
    full[:, valid_mask.to(valid_samples.device)] = valid_samples
    scaled = full * scale.to(valid_samples.device).unsqueeze(0)
    output = scaled.sum(dim=2)
    output = output.permute(0, 1, 3, 2, 4)
    output = output.reshape(
        sample_count,
        output.shape[1] * output.shape[2],
        output.shape[3] * output.shape[4],
    )
    return output[:, :rows, :out_features]
