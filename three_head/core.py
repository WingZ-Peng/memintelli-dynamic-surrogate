from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import torch

from memintelli.pimpy.data_formats import SlicedData
from .observable_dpe import ObservableDPETensor
from .tail import (
    build_valid_mask,
    output_from_valid_s_tile,
    tail_scale,
)

from .features import build_head_features


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def atomic_write_text(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def load_protocol(path: Path, project_root: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    protocol = json.loads(raw.decode("utf-8"))
    fingerprint = hashlib.sha256(raw).hexdigest()
    if protocol.get("schema_version") != 1:
        raise ValueError("Unsupported protocol schema")
    if protocol.get("status") != "pilot_ready":
        raise ValueError("Protocol is not pilot_ready")
    validate_protocol(protocol, project_root)
    return protocol, fingerprint


def validate_protocol(protocol: Mapping[str, Any], project_root: Path) -> None:
    source = protocol["source"]
    for key, hash_key in (
        ("fixed_case", "fixed_case_sha256"),
        ("fixed_contract", "fixed_contract_sha256"),
        ("implementation_manifest", "implementation_manifest_sha256"),
    ):
        path = (project_root / source[key]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != source[hash_key]:
            raise ValueError(f"Source hash mismatch: {path}")

    axes = protocol["fixed_axes"]
    if float(axes["write_variation_in_target"]) != 0.0:
        raise ValueError("Dynamic-only target requires zero write variation")
    if any(float(value) != 0.0 for value in axes["stuck_fault_rates_in_target"]):
        raise ValueError("Dynamic-only target requires zero stuck-fault rates")
    expected_dimensions = {"mean": 138, "variance": 138, "correlation": 118}
    for head, expected in expected_dimensions.items():
        if int(protocol["feature_contracts"][head]["dimensions"]) != expected:
            raise ValueError(f"Unexpected {head} feature dimension")

    seed_ranges: list[tuple[int, int, str]] = []
    stride = int(protocol["seed_contract"]["exact_context_stride"])
    input_seeds: list[int] = []
    for split_name in ("train", "validation", "test"):
        split = protocol["splits"][split_name]
        count = int(split["input_count"])
        samples = int(split["samples_per_context"])
        shard = int(split["contexts_per_shard"])
        if count < 1 or samples < 2 or shard < 1 or samples >= stride:
            raise ValueError(f"Invalid split sizes for {split_name}")
        input_seeds.append(int(split["input_seed"]))
        start = int(split["exact_seed_base"])
        stop = start + (count - 1) * stride + samples - 1
        seed_ranges.append((start, stop, split_name))
    if len(set(input_seeds)) != len(input_seeds):
        raise ValueError("Input seeds overlap")
    for index, (start, stop, name) in enumerate(seed_ranges):
        for other_start, other_stop, other_name in seed_ranges[index + 1 :]:
            if max(start, other_start) <= min(stop, other_stop):
                raise ValueError(f"Exact seed ranges overlap: {name}/{other_name}")

    evaluation = protocol["evaluation"]
    test_samples = int(protocol["splits"]["test"]["samples_per_context"])
    if int(evaluation["exact_candidate_samples"]) + int(
        evaluation["exact_reference_samples"]
    ) != test_samples:
        raise ValueError("Test Exact split differs from samples_per_context")
    if int(evaluation["surrogate_samples"]) != int(
        evaluation["exact_reference_samples"]
    ):
        raise ValueError("Surrogate/reference sample counts differ")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    return device


def set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def engine_configuration(
    case_configuration: Mapping[str, Any],
    device: torch.device,
    *,
    dynamic: bool,
) -> dict[str, Any]:
    result = dict(case_configuration["engine"])
    result["device"] = device
    result["write_variation"] = 0.0
    result["rate_stuck_HGS"] = 0.0
    result["rate_stuck_LGS"] = 0.0
    if not dynamic:
        result["read_variation"] = 0.0
        result["vnoise"] = 0.0
    return result


def new_sliced_data(
    slice_method: torch.Tensor,
    is_weight: bool,
    parallel_size: tuple[int, int],
    device: torch.device,
) -> SlicedData:
    return SlicedData(
        slice_method,
        is_weight=is_weight,
        paral_size=parallel_size,
        quant_gran=parallel_size,
        device=device,
    )


def load_fixed_case(
    protocol: Mapping[str, Any], project_root: Path
) -> dict[str, Any]:
    case = torch.load(
        (project_root / protocol["source"]["fixed_case"]).resolve(),
        map_location="cpu",
        weights_only=False,
    )
    config = case["configuration"]
    axes = protocol["fixed_axes"]
    for key in (
        "input_shape",
        "weight_shape",
        "parallel_size",
        "slice_method",
        "s_tile_shape",
    ):
        if list(config[key]) != list(axes[key]):
            raise ValueError(f"Fixed case {key} differs from protocol")
    return case


def prepare_ideal_weight(
    case: Mapping[str, Any], device: torch.device
) -> tuple[ObservableDPETensor, SlicedData, torch.Tensor]:
    config = case["configuration"]
    engine = ObservableDPETensor(
        **engine_configuration(config, device, dynamic=True)
    )
    slice_method = torch.tensor(config["slice_method"], device=device)
    parallel_size = tuple(config["parallel_size"])
    weight_sliced = new_sliced_data(
        slice_method, True, parallel_size, device
    )
    weight_sliced.slice_data_imp(engine, case["weight"].to(device))
    ideal_g = weight_sliced.G.detach().clone()
    return engine, weight_sliced, ideal_g


def make_input_bank(
    count: int, shape: Sequence[int], seed: int
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(count, *shape, generator=generator)


def output_root(protocol: Mapping[str, Any], experiment_root: Path) -> Path:
    """All consolidated artifacts live under the protocol's own output root, so
    this pipeline never writes into the pinned v1 `outputs/` tree."""
    return experiment_root / protocol["execution"]["output_root"]


def expected_shard_paths(
    protocol: Mapping[str, Any], experiment_root: Path, split_name: str
) -> list[tuple[int, int, Path]]:
    split = protocol["splits"][split_name]
    count = int(split["input_count"])
    width = int(split["contexts_per_shard"])
    directory = output_root(protocol, experiment_root) / "context_shards" / split_name
    return [
        (
            start,
            min(start + width, count),
            directory / f"contexts_{start:04d}_{min(start + width, count) - 1:04d}.pt",
        )
        for start in range(0, count, width)
    ]


def exact_seed(
    protocol: Mapping[str, Any],
    split_name: str,
    context_index: int,
    sample_index: int,
) -> int:
    return (
        int(protocol["splits"][split_name]["exact_seed_base"])
        + context_index
        * int(protocol["seed_contract"]["exact_context_stride"])
        + sample_index
    )


def phase_coefficients(
    input_sliced: SlicedData,
    weight_sliced: SlicedData,
    engine: ObservableDPETensor,
) -> torch.Tensor:
    input_max = input_sliced.sliced_max_weights.to(engine.device)
    weight_max = weight_sliced.sliced_max_weights.to(engine.device)
    shifts = torch.outer(
        input_sliced.sliced_weights.to(engine.device),
        weight_sliced.sliced_weights.to(engine.device),
    )
    adc_ref = (
        (engine.HGS - engine.LGS)
        * engine.vread
        * input_sliced.sliced_data.shape[-1]
    )
    return (
        torch.outer(input_max, weight_max)
        * shifts
        / engine.Q_G
        / engine.vread
        / (engine.g_level - 1)
        * adc_ref
    )


def collect_context_shard(
    *,
    protocol: Mapping[str, Any],
    protocol_fingerprint: str,
    split_name: str,
    start_context: int,
    stop_context: int,
    inputs: torch.Tensor,
    fixed_case: Mapping[str, Any],
    ideal_g: torch.Tensor,
    output_path: Path,
    device: torch.device,
) -> str:
    input_block = inputs[start_context:stop_context]
    if output_path.exists():
        existing = torch.load(output_path, map_location="cpu", weights_only=False)
        expected = (
            existing.get("complete") is True
            and existing.get("protocol_fingerprint") == protocol_fingerprint
            and existing.get("split") == split_name
            and int(existing.get("start_context", -1)) == start_context
            and int(existing.get("stop_context", -1)) == stop_context
            and torch.equal(existing.get("inputs"), input_block)
        )
        if not expected:
            raise ValueError(f"Non-identical existing shard: {output_path}")
        return "reused"

    config = fixed_case["configuration"]
    exact_engine = ObservableDPETensor(
        **engine_configuration(config, device, dynamic=True)
    )
    nominal_engine = ObservableDPETensor(
        **engine_configuration(config, device, dynamic=False)
    )
    slice_method = torch.tensor(config["slice_method"], device=device)
    parallel_size = tuple(config["parallel_size"])
    weight_sliced = new_sliced_data(
        slice_method, True, parallel_size, device
    )
    weight_sliced.slice_data_imp(exact_engine, fixed_case["weight"].to(device))
    if not torch.equal(weight_sliced.G, ideal_g):
        raise RuntimeError("Ideal conductance changed while preparing a shard")

    rows, _ = config["input_shape"]
    out_features = int(config["weight_shape"][1])
    s_tile_shape = tuple(config["s_tile_shape"])
    valid_mask = build_valid_mask(
        config["input_shape"], config["weight_shape"], s_tile_shape, device=device
    )
    sample_count = int(protocol["splits"][split_name]["samples_per_context"])
    exact_s: list[torch.Tensor] = []
    exact_y: list[torch.Tensor] = []
    nominal_s: list[torch.Tensor] = []
    nominal_y: list[torch.Tensor] = []
    nominal_v: list[torch.Tensor] = []
    nominal_pre: list[torch.Tensor] = []
    nominal_post: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    elapsed_ms: list[float] = []
    coefficient: torch.Tensor | None = None
    tail_error = 0.0
    nominal_tail_error = 0.0
    nominal_uses_ideal_g = True

    with torch.inference_mode():
        for local_index, input_cpu in enumerate(input_block):
            global_index = start_context + local_index
            started = perf_counter()
            input_data = input_cpu.to(device)
            input_sliced = new_sliced_data(
                slice_method, False, parallel_size, device
            )
            input_sliced.slice_data_imp(exact_engine, input_data)
            current_coefficient = phase_coefficients(
                input_sliced, weight_sliced, exact_engine
            )
            if coefficient is None:
                coefficient = current_coefficient.detach().clone()
            elif not torch.equal(coefficient, current_coefficient):
                raise RuntimeError("Phase coefficients changed between contexts")

            nominal_output = nominal_engine.MapReduceDot(
                input_sliced, weight_sliced
            )
            trace = nominal_engine.last_trace
            if trace is None:
                raise RuntimeError("Nominal observer produced no trace")
            nominal_uses_ideal_g = nominal_uses_ideal_g and torch.equal(
                trace.g_read, ideal_g
            )
            nominal_valid = trace.s_tile[valid_mask]
            context_s = torch.empty(
                sample_count,
                nominal_valid.numel(),
                dtype=input_data.dtype,
                device=device,
            )
            context_y = torch.empty(
                sample_count,
                rows,
                out_features,
                dtype=input_data.dtype,
                device=device,
            )
            for sample_index in range(sample_count):
                set_seed(
                    exact_seed(
                        protocol, split_name, global_index, sample_index
                    ),
                    device,
                )
                output = exact_engine.MapReduceDot(input_sliced, weight_sliced)
                exact_trace = exact_engine.last_trace
                if exact_trace is None:
                    raise RuntimeError("Exact observer produced no trace")
                context_s[sample_index] = exact_trace.s_tile[valid_mask]
                context_y[sample_index] = output

            current_scale = tail_scale(input_sliced, weight_sliced)
            if local_index == 0:
                reconstructed = output_from_valid_s_tile(
                    context_s,
                    valid_mask,
                    s_tile_shape,
                    current_scale,
                    rows,
                    out_features,
                )
                tail_error = float((reconstructed - context_y).abs().max().item())
                nominal_reconstructed = output_from_valid_s_tile(
                    nominal_valid.unsqueeze(0),
                    valid_mask,
                    s_tile_shape,
                    current_scale,
                    rows,
                    out_features,
                )[0]
                nominal_tail_error = float(
                    (nominal_reconstructed - nominal_output).abs().max().item()
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            exact_s.append(context_s.cpu())
            exact_y.append(context_y.cpu())
            nominal_s.append(nominal_valid.cpu())
            nominal_y.append(nominal_output.cpu())
            nominal_v.append(trace.v_read.cpu())
            nominal_pre.append(trace.pre_adc.cpu())
            nominal_post.append(trace.post_adc_normalized.cpu())
            scales.append(current_scale.cpu())
            elapsed_ms.append((perf_counter() - started) * 1000.0)

    if coefficient is None:
        raise RuntimeError("Empty context shard")
    target_configuration = dict(config)
    target_engine_configuration = dict(config["engine"])
    target_engine_configuration["write_variation"] = 0.0
    target_engine_configuration["rate_stuck_HGS"] = 0.0
    target_engine_configuration["rate_stuck_LGS"] = 0.0
    target_configuration["engine"] = target_engine_configuration
    payload: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "protocol_fingerprint": protocol_fingerprint,
        "split": split_name,
        "start_context": start_context,
        "stop_context": stop_context,
        "inputs": input_block,
        "weight": fixed_case["weight"].cpu(),
        "ideal_g": ideal_g.cpu(),
        "ideal_g_sha256": tensor_sha256(ideal_g),
        "valid_mask": valid_mask.cpu(),
        "s_tile_shape": s_tile_shape,
        "phase_coefficients": coefficient.cpu(),
        "tail_scales": torch.stack(scales),
        "nominal_s_tile": torch.stack(nominal_s),
        "nominal_v_quant": torch.stack(nominal_v),
        "nominal_pre_adc": torch.stack(nominal_pre),
        "nominal_post_adc": torch.stack(nominal_post),
        "nominal_output": torch.stack(nominal_y),
        "conductance_levels": exact_engine.conductance_levels.detach().cpu(),
        "s_tile_valid_samples": torch.stack(exact_s),
        "linear_output_samples": torch.stack(exact_y),
        "configuration": target_configuration,
        "elapsed_context_ms": elapsed_ms,
        "checks": {
            "write_variation_disabled": exact_engine.write_variation == 0.0,
            "stuck_faults_disabled": exact_engine.rate_stuck_HGS == 0.0
            and exact_engine.rate_stuck_LGS == 0.0,
            "nominal_uses_ideal_g": nominal_uses_ideal_g,
            "tail_reconstruction_max_abs_error": tail_error,
            "tail_reconstructs_output": tail_error <= 1e-5,
            "nominal_tail_reconstruction_max_abs_error": nominal_tail_error,
            "nominal_tail_reconstructs_output": nominal_tail_error <= 1e-5,
        },
    }
    payload.update(build_head_features(payload))
    for head_name in ("mean", "variance", "correlation"):
        key = f"{head_name}_features"
        expected = int(protocol["feature_contracts"][head_name]["dimensions"])
        if payload[key].shape != (
            stop_context - start_context,
            int(protocol["fixed_axes"]["valid_s_tile_coordinates"]),
            expected,
        ):
            raise RuntimeError(f"Unexpected {key} shape: {payload[key].shape}")
        if not bool(torch.isfinite(payload[key]).all()):
            raise RuntimeError(f"Non-finite {key}")
    if not all(
        value for value in payload["checks"].values() if isinstance(value, bool)
    ):
        raise RuntimeError(f"Collection checks failed: {payload['checks']}")
    atomic_torch_save(payload, output_path)
    return "written"


def collect_all(
    protocol: Mapping[str, Any],
    protocol_fingerprint: str,
    project_root: Path,
    experiment_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    fixed_case = load_fixed_case(protocol, project_root)
    _, _, ideal_g = prepare_ideal_weight(fixed_case, device)
    records: list[dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        split = protocol["splits"][split_name]
        inputs = make_input_bank(
            int(split["input_count"]),
            protocol["fixed_axes"]["input_shape"],
            int(split["input_seed"]),
        )
        paths = expected_shard_paths(protocol, experiment_root, split_name)
        for shard_index, (start, stop, path) in enumerate(paths):
            started = perf_counter()
            status = collect_context_shard(
                protocol=protocol,
                protocol_fingerprint=protocol_fingerprint,
                split_name=split_name,
                start_context=start,
                stop_context=stop,
                inputs=inputs,
                fixed_case=fixed_case,
                ideal_g=ideal_g,
                output_path=path,
                device=device,
            )
            seconds = perf_counter() - started
            records.append(
                {
                    "split": split_name,
                    "shard_index": shard_index,
                    "start_context": start,
                    "stop_context": stop,
                    "status": status,
                    "path": str(path.resolve()),
                    "seconds": seconds,
                }
            )
            print(
                f"split={split_name} shard={shard_index + 1}/{len(paths)} "
                f"contexts={start}:{stop} status={status} seconds={seconds:.1f}",
                flush=True,
            )
    manifest = {
        "schema_version": 1,
        "complete": True,
        "protocol_fingerprint": protocol_fingerprint,
        "ideal_g_sha256": tensor_sha256(ideal_g),
        "records": records,
    }
    atomic_write_text(
        json.dumps(manifest, indent=2),
        output_root(protocol, experiment_root) / "collection_manifest.json",
    )
    return manifest


def load_split(
    protocol: Mapping[str, Any],
    protocol_fingerprint: str,
    experiment_root: Path,
    split_name: str,
) -> dict[str, Any]:
    shards: list[dict[str, Any]] = []
    ideal_hash: str | None = None
    for start, stop, path in expected_shard_paths(
        protocol, experiment_root, split_name
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if shard.get("complete") is not True:
            raise ValueError(f"Incomplete shard: {path}")
        if shard.get("protocol_fingerprint") != protocol_fingerprint:
            raise ValueError(f"Protocol mismatch: {path}")
        if shard.get("split") != split_name:
            raise ValueError(f"Split mismatch: {path}")
        if int(shard.get("start_context", -1)) != start or int(
            shard.get("stop_context", -1)
        ) != stop:
            raise ValueError(f"Context range mismatch: {path}")
        if ideal_hash is None:
            ideal_hash = str(shard["ideal_g_sha256"])
        elif str(shard["ideal_g_sha256"]) != ideal_hash:
            raise ValueError("Ideal conductance changed between shards")
        if not all(
            value for value in shard["checks"].values() if isinstance(value, bool)
        ):
            raise ValueError(f"Failed collection check: {path}")
        shards.append(shard)

    concatenate_keys = (
        "inputs",
        "tail_scales",
        "nominal_s_tile",
        "nominal_output",
        "mean_features",
        "variance_features",
        "correlation_features",
        "s_tile_valid_samples",
        "linear_output_samples",
    )
    result: dict[str, Any] = {
        key: torch.cat([shard[key] for shard in shards], dim=0)
        for key in concatenate_keys
    }
    result.update(
        {
            "split": split_name,
            "context_count": result["inputs"].shape[0],
            "ideal_g": shards[0]["ideal_g"],
            "ideal_g_sha256": ideal_hash,
            "valid_mask": shards[0]["valid_mask"],
            "s_tile_shape": tuple(shards[0]["s_tile_shape"]),
            "configuration": shards[0]["configuration"],
            "paths": [
                str(path.resolve())
                for _, _, path in expected_shard_paths(
                    protocol, experiment_root, split_name
                )
            ],
        }
    )
    expected_contexts = int(protocol["splits"][split_name]["input_count"])
    expected_samples = int(
        protocol["splits"][split_name]["samples_per_context"]
    )
    if result["context_count"] != expected_contexts:
        raise ValueError("Loaded context count differs from protocol")
    if result["s_tile_valid_samples"].shape[1] != expected_samples:
        raise ValueError("Loaded sample count differs from protocol")
    return result
