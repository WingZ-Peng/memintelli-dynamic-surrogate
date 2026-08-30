from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Optional

import torch

from memintelli.pimpy.data_formats import SlicedData
from memintelli.pimpy.memmat_tensor import DPETensor
from memintelli.pimpy.utils import dot_high_dim


@dataclass
class ExactTrace:
    """Captured tensors at the proposed ExactDynamicBlock boundary."""

    v_read: torch.Tensor
    g_read: torch.Tensor
    pre_adc: torch.Tensor
    post_adc_normalized: torch.Tensor
    s_tile: torch.Tensor
    scaled_tile: torch.Tensor
    output: torch.Tensor
    timings_ms: Dict[str, float]


def _snapshot(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone()


class ObservableDPETensor(DPETensor):
    """DPETensor with a trace for the two-dimensional path.

    The arithmetic mirrors DPETensor._dot() in the source repository. Version 1
    intentionally excludes the three-dimensional Conv/batch branch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_trace: Optional[ExactTrace] = None

    def _dot(self, x: SlicedData, mat: SlicedData, _num2V_func, _num2R_func):
        if len(x.shape) != 2:
            raise NotImplementedError(
                "validation_v1 observes only the two-dimensional Linear path"
            )

        timings_ms: Dict[str, float] = {}

        started = perf_counter()
        v_read = _num2V_func(x)
        timings_ms["voltage_mapping_and_noise"] = (perf_counter() - started) * 1e3

        started = perf_counter()
        g_read = _num2R_func(mat)
        timings_ms["read_variation"] = (perf_counter() - started) * 1e3

        if max(mat.sliced_max_weights) > self.g_level - 1:
            raise ValueError("The weight data is out of the range!")

        adc_ref = (self.HGS - self.LGS) * self.vread * v_read.shape[-1]
        qg = (self.HGS - self.LGS) / (self.g_level - 1)

        started = perf_counter()
        pre_adc = dot_high_dim(v_read, g_read - self.LGS)
        timings_ms["high_dim_mvm"] = (perf_counter() - started) * 1e3

        started = perf_counter()
        if self.radc_is_list:
            radc_expanded = self.radc.view(1, 1, 1, 1, -1, 1, 1)
            post_adc = (
                torch.round(pre_adc / adc_ref * (radc_expanded - 1))
                / (radc_expanded - 1)
            )
        else:
            post_adc = (
                torch.round(pre_adc / adc_ref * (self.radc - 1))
                / (self.radc - 1)
            )
        timings_ms["adc"] = (perf_counter() - started) * 1e3

        started = perf_counter()
        out = torch.mul(
            post_adc,
            x.sliced_max_weights.reshape(1, 1, 1, -1, 1, 1, 1),
        )
        out = (
            torch.mul(
                out,
                mat.sliced_max_weights.reshape(1, 1, 1, 1, -1, 1, 1),
            )
            / qg
            / self.vread
            / (self.g_level - 1)
            * adc_ref
        )

        shift_weights = torch.zeros((len(x), len(mat)), device=x.device)
        for index in range(len(x)):
            shift_weights[index] = x.sliced_weights[index] * mat.sliced_weights

        out = torch.mul(
            out.reshape(
                out.shape[0],
                out.shape[1],
                out.shape[2],
                -1,
                out.shape[5],
                out.shape[6],
            ),
            shift_weights.reshape(1, 1, 1, -1, 1, 1),
        )
        s_tile = out.sum(dim=3)
        timings_ms["slice_rescale_and_reduction"] = (perf_counter() - started) * 1e3

        started = perf_counter()
        if x.bw_e is None:
            out_block_max = torch.einsum("nmij, mpij->nmpij", x.max_data, mat.max_data)
            scaled_tile = (
                s_tile
                * out_block_max
                / (2 ** (sum(x.slice_method) - 1) - 1)
                / (2 ** (sum(mat.slice_method) - 1) - 1)
            )
        else:
            out_block_e_bias = torch.einsum(
                "nmij, mpij->nmpij", 2.0**x.e_bias, 2.0**mat.e_bias
            )
            scaled_tile = (
                s_tile
                * out_block_e_bias
                * 2.0 ** (4 - sum(x.slice_method) - sum(mat.slice_method))
            )
        timings_ms["block_scaling"] = (perf_counter() - started) * 1e3

        started = perf_counter()
        output = scaled_tile.sum(dim=1)
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(
            output.shape[0] * output.shape[1],
            output.shape[2] * output.shape[3],
        )
        output = output[: x.shape[0], : mat.shape[1]]
        timings_ms["k_tile_reduction_and_layout"] = (
            perf_counter() - started
        ) * 1e3

        self.last_trace = ExactTrace(
            v_read=_snapshot(v_read),
            g_read=_snapshot(g_read),
            pre_adc=_snapshot(pre_adc),
            post_adc_normalized=_snapshot(post_adc),
            s_tile=_snapshot(s_tile),
            scaled_tile=_snapshot(scaled_tile),
            output=_snapshot(output),
            timings_ms=timings_ms,
        )
        return output
