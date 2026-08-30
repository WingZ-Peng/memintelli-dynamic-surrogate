from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .conditional_mean import TileConditionalMean
from .core import atomic_write_text, load_split, output_root
from .metrics import (
    correlation_frobenius_offdiagonal_per_context,
    correlation_frobenius_per_context,
    correlation_frobenius_support_per_context,
    empirical_variance,
    mean_nrmse_per_context,
    variance_l1_per_context,
)
from .structure import (
    analytic_anchor,
    analytic_correlation,
    build_coordinate_structure,
    empirical_correlation,
)
from .structured_correlation import StructuredSourceCorrelation, structured_sample
from .tail import output_from_valid_s_tile
from .training import predict_variance


EPS = 1e-12


def aggregate(values: Sequence[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(values), "min": min(values), "max": max(values)}


def covariance(samples: torch.Tensor) -> torch.Tensor:
    centered = samples - samples.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(samples.shape[0] - 1, 1)


def sliced_wasserstein(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    projection_count: int,
    seed: int,
) -> float:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = torch.randn(
        candidate.shape[1], projection_count, dtype=candidate.dtype, generator=generator
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(EPS)
    candidate_projection = torch.sort(candidate @ directions, dim=0).values
    reference_projection = torch.sort(reference @ directions, dim=0).values
    scale = reference.var(dim=0, unbiased=True).mean().sqrt().clamp_min(EPS)
    return float(
        ((candidate_projection - reference_projection).abs().mean() / scale).item()
    )


def distribution_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    quantile_values: Sequence[float],
    projection_count: int,
    projection_seed: int,
) -> dict[str, float]:
    if candidate.shape != reference.shape or candidate.ndim != 2:
        raise ValueError("Distribution samples must share [N,D] shape")
    candidate = candidate.double()
    reference = reference.double()
    candidate_variance = candidate.var(dim=0, unbiased=True)
    reference_variance = reference.var(dim=0, unbiased=True)
    scale = reference_variance.mean().sqrt().clamp_min(EPS)
    quantiles = torch.tensor(tuple(quantile_values), dtype=reference.dtype)
    return {
        "mean_nrmse": float(
            (
                (candidate.mean(dim=0) - reference.mean(dim=0)).square().mean().sqrt()
                / scale
            ).item()
        ),
        "variance_relative_l1": float(
            (
                (candidate_variance - reference_variance).abs().mean()
                / reference_variance.mean().clamp_min(EPS)
            ).item()
        ),
        "covariance_relative_frobenius": float(
            (
                torch.linalg.matrix_norm(covariance(candidate) - covariance(reference))
                / torch.linalg.matrix_norm(covariance(reference)).clamp_min(EPS)
            ).item()
        ),
        "quantile_nrmse": float(
            (
                (
                    torch.quantile(candidate, quantiles, dim=0)
                    - torch.quantile(reference, quantiles, dim=0)
                )
                .square()
                .mean()
                .sqrt()
                / scale
            ).item()
        ),
        "sliced_wasserstein": sliced_wasserstein(
            candidate, reference, projection_count, projection_seed
        ),
    }


def load_models(
    protocol: Mapping[str, Any],
    experiment_root: Path,
    structure_cpu,
    device: torch.device,
) -> tuple[
    TileConditionalMean,
    TileConditionalMean,
    StructuredSourceCorrelation,
    dict[str, torch.Tensor],
]:
    directory = output_root(protocol, experiment_root) / "checkpoints"
    payloads = {
        "mean": torch.load(
            directory / "mean_head.pt", map_location="cpu", weights_only=False
        ),
        "variance": torch.load(
            directory / "variance_head.pt", map_location="cpu", weights_only=False
        ),
        "correlation": torch.load(
            directory / "correlation_head_structured.pt",
            map_location="cpu",
            weights_only=False,
        ),
    }
    for name, payload in payloads.items():
        if payload.get("complete") is not True:
            raise ValueError(f"Incomplete {name} checkpoint")
        if payload["configuration"]["feature_contract"] != dict(
            protocol["feature_contracts"][name]
        ):
            raise ValueError(f"{name} feature contract mismatch")

    target_dim = structure_cpu.dim
    models = {}
    for name in ("mean", "variance"):
        config = protocol["heads"][name]
        model = TileConditionalMean(
            feature_dim=int(protocol["feature_contracts"][name]["dimensions"]),
            target_dim=target_dim,
            hidden_dim=int(config["hidden_dim"]),
            coordinate_dim=int(config["coordinate_dim"]),
        ).to(device)
        model.load_state_dict(payloads[name]["best_model"])
        model.eval()
        models[name] = model

    correlation_config = protocol["heads"]["correlation"]
    correlation_model = StructuredSourceCorrelation(
        feature_dim=int(protocol["feature_contracts"]["correlation"]["dimensions"]),
        structure=structure_cpu,
        hidden_dim=int(correlation_config["hidden_dim"]),
        coordinate_dim=int(correlation_config["coordinate_dim"]),
        source_dim=int(correlation_config["source_dim"]),
        residual_logit_bias=float(correlation_config["residual_logit_bias"]),
        delta_bound=float(correlation_config["delta_bound"]),
    ).to(device)
    correlation_model.load_state_dict(payloads["correlation"]["best_model"])
    correlation_model.eval()

    shared = {
        "mean_delta": payloads["mean"]["global_mean_delta"].float(),
        "variance": payloads["variance"]["global_variance"].float(),
    }
    return models["mean"], models["variance"], correlation_model, shared


def sample_shared_gaussian(
    mean: torch.Tensor,
    variance: torch.Tensor,
    correlation: torch.Tensor,
    sample_count: int,
    seed: int,
) -> torch.Tensor:
    identity = torch.eye(
        correlation.shape[-1], dtype=correlation.dtype, device=correlation.device
    )
    cholesky, info = torch.linalg.cholesky_ex(correlation)
    if int(info.item()) != 0:
        cholesky = torch.linalg.cholesky(correlation + 1e-5 * identity)
    generator = torch.Generator(device=mean.device).manual_seed(seed)
    epsilon = torch.randn(
        mean.shape[0],
        sample_count,
        mean.shape[1],
        generator=generator,
        dtype=mean.dtype,
        device=mean.device,
    )
    return mean.unsqueeze(1) + (epsilon @ cholesky.transpose(-1, -2)) * variance.sqrt().view(
        1, 1, -1
    )


def reconstruct_outputs(samples: torch.Tensor, test: Mapping[str, Any]) -> torch.Tensor:
    rows = int(test["configuration"]["input_shape"][0])
    out_features = int(test["configuration"]["weight_shape"][1])
    return torch.stack(
        [
            output_from_valid_s_tile(
                samples[context_index],
                test["valid_mask"],
                test["s_tile_shape"],
                test["tail_scales"][context_index],
                rows,
                out_features,
            )
            for context_index in range(samples.shape[0])
        ]
    )


def evaluate_distributions(
    protocol: Mapping[str, Any],
    test: Mapping[str, Any],
    methods: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    evaluation = protocol["evaluation"]
    candidate_count = int(evaluation["exact_candidate_samples"])
    reference_count = int(evaluation["exact_reference_samples"])
    exact_y = test["linear_output_samples"].float()
    reference_s = test["s_tile_valid_samples"].float()[
        :, candidate_count : candidate_count + reference_count
    ]
    reference_y = exact_y[:, candidate_count : candidate_count + reference_count]
    method_outputs = {
        name: (
            exact_y[:, :candidate_count]
            if name == "exact_independent_split"
            else reconstruct_outputs(samples, test)
        )
        for name, samples in methods.items()
    }
    result: dict[str, Any] = {}
    for method_name, candidate_s in methods.items():
        s_records, y_records = [], []
        for context_index in range(test["context_count"]):
            projection_seed = 610_000_000 + context_index
            s_records.append(
                distribution_metrics(
                    candidate_s[context_index],
                    reference_s[context_index],
                    evaluation["quantiles"],
                    int(evaluation["sliced_wasserstein_projections"]),
                    projection_seed,
                )
            )
            y_records.append(
                distribution_metrics(
                    method_outputs[method_name][context_index].reshape(
                        candidate_count, -1
                    ),
                    reference_y[context_index].reshape(reference_count, -1),
                    evaluation["quantiles"],
                    int(evaluation["sliced_wasserstein_projections"]),
                    projection_seed + 100_000,
                )
            )
        result[method_name] = {
            "s_tile": {
                metric: aggregate([record[metric] for record in s_records])
                for metric in s_records[0]
            },
            "linear_output": {
                metric: aggregate([record[metric] for record in y_records])
                for metric in y_records[0]
            },
        }
    return result


def structural_summary(reference: torch.Tensor, structure) -> dict[str, Any]:
    eye = torch.eye(structure.dim, dtype=torch.bool)
    same_k = structure.k_block[:, None] == structure.k_block[None, :]
    same_row = structure.row_index[:, None] == structure.row_index[None, :]
    same_out = structure.out_index[:, None] == structure.out_index[None, :]
    groups = {
        "shared_row": same_k & same_row & ~same_out,
        "shared_out": same_k & same_out & ~same_row,
        "same_k_disjoint": same_k & ~same_row & ~same_out & ~eye,
        "cross_k_block": ~same_k,
    }
    total = float(reference[:, ~eye].pow(2).sum())
    return {
        name: {
            "pair_count": int(mask.sum()),
            "rms_rho": float(reference[:, mask].pow(2).mean().sqrt()),
            "share_of_offdiagonal_energy": float(
                reference[:, mask].pow(2).sum() / max(total, EPS)
            ),
        }
        for name, mask in groups.items()
    }


def analyze(
    protocol: Mapping[str, Any],
    protocol_fingerprint: str,
    experiment_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    root = output_root(protocol, experiment_root)
    training_summary = json.loads(
        (root / "training_summary.json").read_text(encoding="utf-8")
    )
    if training_summary.get("protocol_fingerprint") != protocol_fingerprint:
        raise ValueError("Training summary protocol mismatch")
    train = load_split(protocol, protocol_fingerprint, experiment_root, "train")
    test = load_split(protocol, protocol_fingerprint, experiment_root, "test")
    structure_cpu = build_coordinate_structure(
        test["valid_mask"], test["configuration"]
    )
    structure = structure_cpu.to(device)
    dim = structure_cpu.dim

    mean_model, variance_model, correlation_model, shared = load_models(
        protocol, experiment_root, structure_cpu, device
    )
    shared_correlation = empirical_correlation(
        train["s_tile_valid_samples"].float()
    ).mean(dim=0)

    mean_features = test["mean_features"].float().to(device)
    variance_features = test["variance_features"].float().to(device)
    correlation_features = test["correlation_features"].float().to(device)
    nominal = test["nominal_s_tile"].float().to(device)
    shuffled_mean = torch.roll(mean_features, shifts=1, dims=0)
    shuffled_variance = torch.roll(variance_features, shifts=1, dims=0)
    shuffled_correlation = torch.roll(correlation_features, shifts=1, dims=0)

    samples = test["s_tile_valid_samples"].float()
    candidate_count = int(protocol["evaluation"]["exact_candidate_samples"])
    reference_count = int(protocol["evaluation"]["exact_reference_samples"])
    candidate = samples[:, :candidate_count]
    reference = samples[:, candidate_count : candidate_count + reference_count]
    candidate_correlation = empirical_correlation(candidate)
    reference_correlation = empirical_correlation(reference)
    maximum = float(protocol["heads"]["variance"]["maximum_log_variance_ratio"])
    anchor = analytic_anchor(test["correlation_features"].float())
    analytic = analytic_correlation(anchor, structure_cpu)

    with torch.no_grad():
        conditional_mean = nominal + mean_model(mean_features)
        shuffled_mean_prediction = nominal + mean_model(shuffled_mean)
        shared_mean = nominal + shared["mean_delta"].to(device)
        conditional_variance = predict_variance(
            variance_model, variance_features, shared["variance"].to(device), maximum
        )
        shuffled_variance_prediction = predict_variance(
            variance_model, shuffled_variance, shared["variance"].to(device), maximum
        )
        shared_variance = shared["variance"].to(device).unsqueeze(0).expand_as(
            conditional_variance
        )
        conditional_correlation = correlation_model.correlation_matrix(
            correlation_features
        )
        shuffled_correlation_prediction = correlation_model.correlation_matrix(
            shuffled_correlation
        )

    head_values = {
        "mean": {
            "exact_independent_split": mean_nrmse_per_context(
                candidate.mean(dim=1), reference
            ),
            "conditional": mean_nrmse_per_context(conditional_mean.cpu(), reference),
            "shuffled_input_features": mean_nrmse_per_context(
                shuffled_mean_prediction.cpu(), reference
            ),
            "shared": mean_nrmse_per_context(shared_mean.cpu(), reference),
        },
        "variance": {
            "exact_independent_split": variance_l1_per_context(
                empirical_variance(candidate), reference
            ),
            "conditional": variance_l1_per_context(
                conditional_variance.cpu(), reference
            ),
            "shuffled_input_features": variance_l1_per_context(
                shuffled_variance_prediction.cpu(), reference
            ),
            "shared": variance_l1_per_context(shared_variance.cpu(), reference),
        },
    }
    head_metrics = {
        head: {
            method: aggregate([float(value) for value in values])
            for method, values in methods.items()
        }
        for head, methods in head_values.items()
    }

    correlation_predictions = {
        "exact_independent_split": candidate_correlation,
        "identity": torch.eye(dim).unsqueeze(0).expand_as(reference_correlation),
        "shared_empirical": shared_correlation.unsqueeze(0).expand_as(
            reference_correlation
        ),
        "shuffled_input_features": shuffled_correlation_prediction.cpu(),
        "analytic_first_order": analytic,
        "structured_conditional": conditional_correlation.cpu(),
    }
    correlation_metrics = {
        name: {
            "full_matrix_frobenius": aggregate(
                [
                    float(v)
                    for v in correlation_frobenius_per_context(
                        prediction, reference_correlation
                    )
                ]
            ),
            "offdiagonal_frobenius": aggregate(
                [
                    float(v)
                    for v in correlation_frobenius_offdiagonal_per_context(
                        prediction, reference_correlation
                    )
                ]
            ),
            "support_frobenius": aggregate(
                [
                    float(v)
                    for v in correlation_frobenius_support_per_context(
                        prediction, reference_correlation, structure_cpu
                    )
                ]
            ),
        }
        for name, prediction in correlation_predictions.items()
    }
    head_metrics["correlation"] = {
        method: correlation_metrics[key]["support_frobenius"]
        for method, key in (
            ("exact_independent_split", "exact_independent_split"),
            ("conditional", "structured_conditional"),
            ("shuffled_input_features", "shuffled_input_features"),
            ("shared", "shared_empirical"),
        )
    }
    floor_ratio = {
        head: metrics["conditional"]["mean"]
        / metrics["exact_independent_split"]["mean"]
        for head, metrics in head_metrics.items()
    }

    sample_count = int(protocol["evaluation"]["surrogate_samples"])
    seed = int(protocol["seed_contract"]["surrogate_seed_base"])
    source_dim = int(protocol["heads"]["correlation"]["source_dim"])
    anchor_device = anchor.to(device)
    with torch.no_grad():
        conditional_samples = conditional_mean.unsqueeze(1) + correlation_model.sample(
            correlation_features,
            conditional_variance,
            sample_count,
            structure,
            torch.Generator(device=device).manual_seed(seed),
        )
        analytic_samples = conditional_mean.unsqueeze(1) + structured_sample(
            anchor_device[..., :source_dim],
            anchor_device[..., source_dim:],
            torch.zeros_like(conditional_variance),
            structure,
            conditional_variance,
            sample_count,
            torch.Generator(device=device).manual_seed(seed),
        )
        shuffled_samples = shuffled_mean_prediction.unsqueeze(
            1
        ) + correlation_model.sample(
            shuffled_correlation,
            shuffled_variance_prediction,
            sample_count,
            structure,
            torch.Generator(device=device).manual_seed(seed),
        )
        shared_samples = sample_shared_gaussian(
            shared_mean,
            shared["variance"].to(device),
            shared_correlation.to(device),
            sample_count,
            seed,
        )

    methods = {
        "exact_independent_split": candidate,
        "conditional_three_head": conditional_samples.cpu(),
        "analytic_correlation_three_head": analytic_samples.cpu(),
        "shuffled_input_features": shuffled_samples.cpu(),
        "shared_three_head": shared_samples.cpu(),
        "nominal_only": nominal.cpu().unsqueeze(1).expand(-1, sample_count, -1),
    }
    distribution = evaluate_distributions(protocol, test, methods)
    distribution_floor = distribution["exact_independent_split"]
    distribution_ratio = {
        name: {
            space: {
                metric: values["mean"]
                / max(distribution_floor[space][metric]["mean"], EPS)
                for metric, values in metrics.items()
            }
            for space, metrics in spaces.items()
        }
        for name, spaces in distribution.items()
    }

    summary: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "protocol_fingerprint": protocol_fingerprint,
        "scope": protocol["scope"],
        "counts": {
            "train_contexts": train["context_count"],
            "validation_contexts": int(
                protocol["splits"]["validation"]["input_count"]
            ),
            "test_contexts": test["context_count"],
            "train_exact_samples_per_context": int(
                protocol["splits"]["train"]["samples_per_context"]
            ),
            "test_exact_samples_per_context": samples.shape[1],
            "test_surrogate_samples_per_context": sample_count,
        },
        "feature_contracts": protocol["feature_contracts"],
        "structure": structural_summary(reference_correlation, structure_cpu),
        "training": training_summary,
        "head_metrics": head_metrics,
        "conditional_to_exact_floor_ratio": floor_ratio,
        "correlation_metrics": correlation_metrics,
        "distribution": distribution,
        "distribution_to_exact_floor_ratio": distribution_ratio,
        "limitations": [
            "One fixed mathematical weight and ideal conductance tensor.",
            "Write variation and stuck faults are deliberately excluded.",
            "One fixed shape, tile layout, and hardware configuration.",
            "Synthetic Gaussian inputs, not real-network activations.",
            "Head schedules and the correlation regularizer were chosen on the "
            "validation split; the test split is read only here.",
        ],
    }
    atomic_write_text(
        json.dumps(summary, indent=2), root / "report" / "three_head_results.json"
    )
    atomic_write_text(
        build_markdown(summary), root / "report" / "three_head_results.md"
    )
    return summary


def build_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Ideal-Conductance Dynamic Three-Head Surrogate",
        "",
        "Weight, ideal quantized/sliced conductance, layout, and hardware config",
        "are fixed. Write variation and stuck faults are excluded. All test inputs",
        "are held out.",
        "",
        "## Head diagnostics",
        "",
        "Correlation rows are scored on the physical support: the 1,300 of 9,900",
        "off-diagonal entries that can be non-zero. The rest are exactly zero in",
        "truth, so scoring them measures the reference's sampling noise.",
        "",
        "| Metric | Exact floor | Conditional | Shuffled input features | Shared | Floor ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "mean": "Mean NRMSE",
        "variance": "Variance L1",
        "correlation": "Correlation Fro. (support)",
    }
    for head_name in ("mean", "variance", "correlation"):
        metrics = summary["head_metrics"][head_name]
        lines.append(
            f"| {labels[head_name]} | {metrics['exact_independent_split']['mean']:.4f} | "
            f"{metrics['conditional']['mean']:.4f} | "
            f"{metrics['shuffled_input_features']['mean']:.4f} | "
            f"{metrics['shared']['mean']:.4f} | "
            f"{summary['conditional_to_exact_floor_ratio'][head_name]:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Where the correlation lives",
            "",
            "| Pair type | Pairs | RMS rho | Share of off-diagonal energy |",
            "|---|---:|---:|---:|",
        ]
    )
    structure_labels = {
        "shared_row": "same k, shared row",
        "shared_out": "same k, shared output column",
        "same_k_disjoint": "same k, disjoint (exactly 0)",
        "cross_k_block": "different k_block (exactly 0)",
    }
    for key, label in structure_labels.items():
        record = summary["structure"][key]
        lines.append(
            f"| {label} | {record['pair_count']} | {record['rms_rho']:.4f} | "
            f"{100 * record['share_of_offdiagonal_energy']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Correlation predictors",
            "",
            "| Predictor | Full matrix | Off-diagonal | Support only |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, record in summary["correlation_metrics"].items():
        lines.append(
            f"| `{name}` | {record['full_matrix_frobenius']['mean']:.4f} | "
            f"{record['offdiagonal_frobenius']['mean']:.4f} | "
            f"{record['support_frobenius']['mean']:.4f} |"
        )
    lines.extend(["", "## End-to-end distribution", ""])
    for space in ("s_tile", "linear_output"):
        lines.extend(
            [
                f"### {space}",
                "",
                "| Method | Mean NRMSE | Variance L1 | Covariance Fro. | Quantile NRMSE | Sliced W. |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method, spaces in summary["distribution"].items():
            metrics = spaces[space]
            lines.append(
                f"| `{method}` | {metrics['mean_nrmse']['mean']:.4f} | "
                f"{metrics['variance_relative_l1']['mean']:.4f} | "
                f"{metrics['covariance_relative_frobenius']['mean']:.4f} | "
                f"{metrics['quantile_nrmse']['mean']:.4f} | "
                f"{metrics['sliced_wasserstein']['mean']:.4f} |"
            )
        lines.append("")
    lines.extend(
        [
            "### Distance to the Exact finite-sample floor",
            "",
            "| Method | s_tile covariance | linear_output variance | linear_output covariance |",
            "|---|---:|---:|---:|",
        ]
    )
    for method, spaces in summary["distribution_to_exact_floor_ratio"].items():
        if method == "nominal_only":
            continue
        lines.append(
            f"| `{method}` | {spaces['s_tile']['covariance_relative_frobenius']:.3f} | "
            f"{spaces['linear_output']['variance_relative_l1']:.3f} | "
            f"{spaces['linear_output']['covariance_relative_frobenius']:.3f} |"
        )
    lines.extend(["", "## What this does not show", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    return "\n".join(lines) + "\n"
