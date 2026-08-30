from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F

from .conditional_mean import TileConditionalMean
from .core import atomic_torch_save, atomic_write_text, load_split, output_root
from .metrics import (
    correlation_frobenius_offdiagonal_per_context,
    correlation_frobenius_support_per_context,
    empirical_variance,
    gaussian_negative_log_likelihood,
    mean_nrmse_per_context,
    support_relative_squared_loss,
    variance_l1_per_context,
)
from .structure import (
    analytic_anchor,
    build_coordinate_structure,
    empirical_correlation,
)
from .structured_correlation import StructuredSourceCorrelation


EPS = 1e-12


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in module.state_dict().items()
    }


def data_fingerprint(split: Mapping[str, Any]) -> str:
    payload = {
        "split": split["split"],
        "context_count": split["context_count"],
        "ideal_g_sha256": split["ideal_g_sha256"],
        "input_shape": list(split["inputs"].shape),
        "sample_shape": list(split["s_tile_valid_samples"].shape),
        "mean_feature_shape": list(split["mean_features"].shape),
        "variance_feature_shape": list(split["variance_features"].shape),
        "correlation_feature_shape": list(split["correlation_features"].shape),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# --- Previous schedule, from ideal_dynamic/training.py ------------------------
# The v1 trainer held the learning rate fixed for the whole run and evaluated
# validation only every 100 steps:
#
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=float(config["learning_rate"]),        # 1e-3, never decayed
#         weight_decay=float(config["weight_decay"]),
#     )
#     for step in range(start_step, int(config["steps"]) + 1):
#         ...
#         optimizer.step()
#
# The variance head's validation metric then swung between 0.0567 and 0.1948
# across consecutive reports (3.43x) and its selected checkpoint was the final
# step: had the run stopped at step 900 the saved head would have scored 0.1514.
# Labels were not the cause (log-ratio signal-to-noise 10.1x) and neither was the
# metric (standard deviation 0.0005 across random reference halves) -- the
# optimizer was still taking full-size steps at the end of training.
# -----------------------------------------------------------------------------


def learning_rate_scale(
    step: int, total_steps: int, warmup_steps: int, schedule: str
) -> float:
    """Linear warmup, then cosine decay to zero.

    Warmup protects the zero-initialized output layer at the start; the decay is
    what removes the oscillation, by shrinking the step size as the model reaches
    the minimum instead of continuing to bounce across it.
    """
    if step <= warmup_steps:
        return step / max(warmup_steps, 1)
    if schedule == "constant":
        return 1.0
    if schedule != "cosine":
        raise ValueError(f"Unsupported schedule: {schedule}")
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def curve_statistics(history: list[dict[str, float]], total_steps: int) -> dict[str, Any]:
    """A converged run is smooth late and does not peak on its final step."""
    values = [(int(record["step"]), record["validation"]) for record in history]
    late = [value for step, value in values if step > total_steps // 2]
    roughness = (
        statistics.fmean(abs(late[i] - late[i - 1]) for i in range(1, len(late)))
        if len(late) > 1
        else 0.0
    )
    best_value, best_step = min((value, step) for step, value in values)
    return {
        "report_count": len(values),
        "best": best_value,
        "best_step": best_step,
        "best_is_final_step": best_step == total_steps,
        "final": values[-1][1],
        "late_curve_roughness": roughness,
    }


def checkpoint_configuration(
    protocol_fingerprint: str,
    training_fingerprint: str,
    head_name: str,
    head_config: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_fingerprint": protocol_fingerprint,
        "training_data_fingerprint": training_fingerprint,
        "head": head_name,
        "head_config": dict(head_config),
        "feature_contract": dict(feature_contract),
    }


def restore_checkpoint(
    *,
    path: Path,
    configuration: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
) -> tuple[int, int, float, dict[str, torch.Tensor] | None, list[dict[str, float]]]:
    if not path.exists():
        return 1, 0, float("inf"), None, []
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("configuration") != dict(configuration):
        raise ValueError(f"Checkpoint configuration mismatch: {path}")
    model.load_state_dict(payload["current_model"])
    optimizer.load_state_dict(payload["optimizer"])
    generator.set_state(payload["generator_state"])
    return (
        int(payload["step"]) + 1,
        int(payload["best_step"]),
        float(payload["best_validation"]),
        payload["best_model"],
        list(payload["history"]),
    )


def save_checkpoint(
    *,
    path: Path,
    configuration: Mapping[str, Any],
    step: int,
    total_steps: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    best_step: int,
    best_validation: float,
    best_model: Mapping[str, torch.Tensor],
    history: list[dict[str, float]],
    extra: Mapping[str, Any],
) -> None:
    atomic_torch_save(
        {
            "schema_version": 1,
            "configuration": dict(configuration),
            "step": step,
            "complete": step == total_steps,
            "current_model": cpu_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "generator_state": generator.get_state(),
            "best_step": best_step,
            "best_validation": best_validation,
            "best_model": dict(best_model),
            "history": history,
            **dict(extra),
        },
        path,
    )


def train_tile_regression_head(
    *,
    protocol: Mapping[str, Any],
    protocol_fingerprint: str,
    train: Mapping[str, Any],
    head_name: str,
    train_features: torch.Tensor,
    train_target: torch.Tensor,
    validation_metric: Callable[[TileConditionalMean], float],
    output_path: Path,
    device: torch.device,
    checkpoint_extra: Mapping[str, Any],
) -> tuple[TileConditionalMean, dict[str, Any]]:
    config = protocol["heads"][head_name]
    seed_offset = 1 if head_name == "mean" else 2
    seed = int(protocol["seed_contract"]["model_seed_base"]) + seed_offset
    torch.manual_seed(seed)
    model = TileConditionalMean(
        feature_dim=train_features.shape[-1],
        target_dim=train_features.shape[1],
        hidden_dim=int(config["hidden_dim"]),
        coordinate_dim=int(config["coordinate_dim"]),
    ).to(device)
    model.configure_normalization(train_features, train_target)
    base_learning_rate = float(config["learning_rate"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_learning_rate,
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 1000)
    configuration = checkpoint_configuration(
        protocol_fingerprint,
        data_fingerprint(train),
        head_name,
        config,
        protocol["feature_contracts"][head_name],
    )
    start_step, best_step, best_validation, best_model, history = restore_checkpoint(
        path=output_path,
        configuration=configuration,
        model=model,
        optimizer=optimizer,
        generator=generator,
    )
    total_steps = int(config["steps"])
    if start_step > total_steps:
        if best_model is None:
            raise RuntimeError(f"Complete {head_name} checkpoint has no best model")
        model.load_state_dict(best_model)
        model.eval()
        return model, {
            "best_step": best_step,
            "best_validation": best_validation,
            "curve": curve_statistics(history, total_steps),
            "history": history,
            "resumed_complete": True,
        }

    features_device = train_features.to(device)
    target_device = train_target.to(device)
    warmup = int(config["warmup_steps"])
    schedule = str(config["schedule"])
    started = perf_counter()
    for step in range(start_step, total_steps + 1):
        scale = learning_rate_scale(step, total_steps, warmup, schedule)
        for group in optimizer.param_groups:
            group["lr"] = base_learning_rate * scale
        indices = torch.randint(
            train_features.shape[0],
            (min(int(config["batch_contexts"]), train_features.shape[0]),),
            generator=generator,
        ).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, normalized_target = model.prediction_and_normalized_target(
            features_device[indices], target_device[indices]
        )
        loss = F.smooth_l1_loss(
            prediction, normalized_target, beta=float(config.get("huber_beta", 0.5))
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["gradient_clip"])
        )
        optimizer.step()
        report = (
            step == 1 or step % int(config["report_every"]) == 0 or step == total_steps
        )
        if not report:
            continue
        model.eval()
        with torch.no_grad():
            metric = validation_metric(model)
        history.append(
            {
                "step": float(step),
                "loss": float(loss.item()),
                "learning_rate": base_learning_rate * scale,
                "validation": metric,
            }
        )
        if metric < best_validation:
            best_step, best_validation = step, metric
            best_model = cpu_state_dict(model)
        if best_model is None:
            raise RuntimeError(f"{head_name} training produced no best model")
        save_checkpoint(
            path=output_path,
            configuration=configuration,
            step=step,
            total_steps=total_steps,
            model=model,
            optimizer=optimizer,
            generator=generator,
            best_step=best_step,
            best_validation=best_validation,
            best_model=best_model,
            history=history,
            extra={
                **dict(checkpoint_extra),
                "training_seconds_this_run": perf_counter() - started,
            },
        )
        if step == 1 or step % (int(config["report_every"]) * 10) == 0 or step == total_steps:
            print(
                f"head={head_name} step={step}/{total_steps} "
                f"loss={loss.item():.6f} val={metric:.6f} "
                f"best={best_validation:.6f}@{best_step}",
                flush=True,
            )
    if best_model is None:
        raise RuntimeError(f"{head_name} training produced no checkpoint")
    model.load_state_dict(best_model)
    model.eval()
    return model, {
        "best_step": best_step,
        "best_validation": best_validation,
        "curve": curve_statistics(history, total_steps),
        "history": history,
        "resumed_complete": False,
    }


def predict_variance(
    model: TileConditionalMean,
    features: torch.Tensor,
    global_variance: torch.Tensor,
    maximum_log_ratio: float,
) -> torch.Tensor:
    log_ratio = model(features).clamp(-maximum_log_ratio, maximum_log_ratio)
    return global_variance.unsqueeze(0) * log_ratio.exp()


def train_mean_head(
    *,
    protocol: Mapping[str, Any],
    protocol_fingerprint: str,
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    output_path: Path,
    device: torch.device,
) -> tuple[TileConditionalMean, dict[str, Any]]:
    train_samples = train["s_tile_valid_samples"].float()
    target = train_samples.mean(dim=1) - train["nominal_s_tile"].float()
    validation_features = validation["mean_features"].float().to(device)
    validation_nominal = validation["nominal_s_tile"].float().to(device)
    validation_samples = validation["s_tile_valid_samples"].float()
    reference = validation_samples[:, validation_samples.shape[1] // 2 :].to(device)

    def metric(model: TileConditionalMean) -> float:
        prediction = validation_nominal + model(validation_features)
        return float(mean_nrmse_per_context(prediction, reference).mean().item())

    return train_tile_regression_head(
        protocol=protocol,
        protocol_fingerprint=protocol_fingerprint,
        train=train,
        head_name="mean",
        train_features=train["mean_features"].float(),
        train_target=target,
        validation_metric=metric,
        output_path=output_path,
        device=device,
        checkpoint_extra={"global_mean_delta": target.mean(dim=0)},
    )


def train_variance_head(
    *,
    protocol: Mapping[str, Any],
    protocol_fingerprint: str,
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    output_path: Path,
    device: torch.device,
) -> tuple[TileConditionalMean, torch.Tensor, dict[str, Any]]:
    train_variance = empirical_variance(train["s_tile_valid_samples"].float())
    global_variance = train_variance.mean(dim=0).clamp_min(EPS)
    maximum = float(protocol["heads"]["variance"]["maximum_log_variance_ratio"])
    target = (train_variance / global_variance).log().clamp(-maximum, maximum)
    validation_features = validation["variance_features"].float().to(device)
    validation_samples = validation["s_tile_valid_samples"].float()
    reference = validation_samples[:, validation_samples.shape[1] // 2 :].to(device)
    global_device = global_variance.to(device)

    def metric(model: TileConditionalMean) -> float:
        prediction = predict_variance(
            model, validation_features, global_device, maximum
        )
        return float(variance_l1_per_context(prediction, reference).mean().item())

    model, summary = train_tile_regression_head(
        protocol=protocol,
        protocol_fingerprint=protocol_fingerprint,
        train=train,
        head_name="variance",
        train_features=train["variance_features"].float(),
        train_target=target,
        validation_metric=metric,
        output_path=output_path,
        device=device,
        checkpoint_extra={"global_variance": global_variance},
    )
    return model, global_variance, summary


def train_correlation_head(
    *,
    protocol: Mapping[str, Any],
    protocol_fingerprint: str,
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    output_path: Path,
    device: torch.device,
) -> tuple[StructuredSourceCorrelation, dict[str, Any]]:
    config = protocol["heads"]["correlation"]
    structure_cpu = build_coordinate_structure(
        train["valid_mask"], train["configuration"]
    )
    structure = structure_cpu.to(device)
    train_features = train["correlation_features"].float()
    # Raw empirical correlations, no shrinkage toward identity: the structured
    # head predicts exact zeros off the physical support, so the noise there
    # carries no gradient, and the loss is restricted to the support instead.
    train_target = empirical_correlation(train["s_tile_valid_samples"].float())
    validation_features = validation["correlation_features"].float().to(device)
    validation_samples = validation["s_tile_valid_samples"].float()
    validation_reference = empirical_correlation(
        validation_samples[:, validation_samples.shape[1] // 2 :]
    ).to(device)

    seed = int(protocol["seed_contract"]["model_seed_base"]) + 3
    torch.manual_seed(seed)
    model = StructuredSourceCorrelation(
        feature_dim=train_features.shape[-1],
        structure=structure_cpu,
        hidden_dim=int(config["hidden_dim"]),
        coordinate_dim=int(config["coordinate_dim"]),
        source_dim=int(config["source_dim"]),
        residual_logit_bias=float(config["residual_logit_bias"]),
        delta_bound=float(config["delta_bound"]),
    ).to(device)
    model.configure_normalization(train_features)
    base_learning_rate = float(config["learning_rate"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_learning_rate,
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 1000)
    configuration = checkpoint_configuration(
        protocol_fingerprint,
        data_fingerprint(train),
        "correlation",
        config,
        protocol["feature_contracts"]["correlation"],
    )
    start_step, best_step, best_validation, best_model, history = restore_checkpoint(
        path=output_path,
        configuration=configuration,
        model=model,
        optimizer=optimizer,
        generator=generator,
    )
    total_steps = int(config["steps"])
    if start_step > total_steps:
        if best_model is None:
            raise RuntimeError("Complete correlation checkpoint has no best model")
        model.load_state_dict(best_model)
        model.eval()
        return model, {
            "best_step": best_step,
            "best_validation": best_validation,
            "curve": curve_statistics(history, total_steps),
            "history": history,
            "resumed_complete": True,
        }

    features_device = train_features.to(device)
    targets_device = train_target.to(device)
    warmup = int(config["warmup_steps"])
    schedule = str(config["schedule"])
    started = perf_counter()
    for step in range(start_step, total_steps + 1):
        scale = learning_rate_scale(step, total_steps, warmup, schedule)
        for group in optimizer.param_groups:
            group["lr"] = base_learning_rate * scale
        indices = torch.randint(
            train_features.shape[0],
            (min(int(config["batch_contexts"]), train_features.shape[0]),),
            generator=generator,
        ).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, deviation = model.correlation_matrix_and_deviation(
            features_device[indices]
        )
        target = targets_device[indices]
        matrix_loss = support_relative_squared_loss(prediction, target, structure)
        likelihood = gaussian_negative_log_likelihood(prediction, target)
        anchor_penalty = deviation.square().mean()
        loss = (
            float(config["matrix_weight"]) * matrix_loss
            + float(config["negative_log_likelihood_weight"]) * likelihood
            + float(config["anchor_penalty_weight"]) * anchor_penalty
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["gradient_clip"])
        )
        optimizer.step()
        report = (
            step == 1 or step % int(config["report_every"]) == 0 or step == total_steps
        )
        if not report:
            continue
        model.eval()
        with torch.no_grad():
            validation_prediction = model.correlation_matrix(validation_features)
            metric = float(
                correlation_frobenius_offdiagonal_per_context(
                    validation_prediction, validation_reference
                )
                .mean()
                .item()
            )
            support_metric = float(
                correlation_frobenius_support_per_context(
                    validation_prediction, validation_reference, structure
                )
                .mean()
                .item()
            )
        history.append(
            {
                "step": float(step),
                "loss": float(loss.item()),
                "matrix_loss": float(matrix_loss.item()),
                "negative_log_likelihood": float(likelihood.item()),
                "anchor_penalty": float(anchor_penalty.item()),
                "learning_rate": base_learning_rate * scale,
                "validation": metric,
                "validation_support": support_metric,
            }
        )
        if metric < best_validation:
            best_step, best_validation = step, metric
            best_model = cpu_state_dict(model)
        if best_model is None:
            raise RuntimeError("Correlation training produced no best model")
        save_checkpoint(
            path=output_path,
            configuration=configuration,
            step=step,
            total_steps=total_steps,
            model=model,
            optimizer=optimizer,
            generator=generator,
            best_step=best_step,
            best_validation=best_validation,
            best_model=best_model,
            history=history,
            extra={"training_seconds_this_run": perf_counter() - started},
        )
        if step == 1 or step % (int(config["report_every"]) * 10) == 0 or step == total_steps:
            print(
                f"head=correlation step={step}/{total_steps} loss={loss.item():.6f} "
                f"val_offdiag={metric:.6f} val_support={support_metric:.6f} "
                f"best={best_validation:.6f}@{best_step}",
                flush=True,
            )
    if best_model is None:
        raise RuntimeError("Correlation training produced no checkpoint")
    model.load_state_dict(best_model)
    model.eval()
    return model, {
        "best_step": best_step,
        "best_validation": best_validation,
        "curve": curve_statistics(history, total_steps),
        "history": history,
        "resumed_complete": False,
    }


def train_all(
    protocol: Mapping[str, Any],
    protocol_fingerprint: str,
    experiment_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    train = load_split(protocol, protocol_fingerprint, experiment_root, "train")
    validation = load_split(
        protocol, protocol_fingerprint, experiment_root, "validation"
    )
    directory = output_root(protocol, experiment_root) / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    _, mean_summary = train_mean_head(
        protocol=protocol,
        protocol_fingerprint=protocol_fingerprint,
        train=train,
        validation=validation,
        output_path=directory / "mean_head.pt",
        device=device,
    )
    _, _, variance_summary = train_variance_head(
        protocol=protocol,
        protocol_fingerprint=protocol_fingerprint,
        train=train,
        validation=validation,
        output_path=directory / "variance_head.pt",
        device=device,
    )
    _, correlation_summary = train_correlation_head(
        protocol=protocol,
        protocol_fingerprint=protocol_fingerprint,
        train=train,
        validation=validation,
        output_path=directory / "correlation_head_structured.pt",
        device=device,
    )
    summary = {
        "schema_version": 1,
        "complete": True,
        "protocol_fingerprint": protocol_fingerprint,
        "training_data_fingerprint": data_fingerprint(train),
        "train_contexts": train["context_count"],
        "validation_contexts": validation["context_count"],
        "heads": {
            "mean": mean_summary,
            "variance": variance_summary,
            "correlation": correlation_summary,
        },
    }
    atomic_write_text(
        json.dumps(summary, indent=2),
        output_root(protocol, experiment_root) / "training_summary.json",
    )
    return summary
