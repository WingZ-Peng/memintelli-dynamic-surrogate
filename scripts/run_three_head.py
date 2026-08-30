from __future__ import annotations

import argparse
import sys
from pathlib import Path


# In this standalone repository the experiment root and the project root are the
# same directory. The protocol resolves its pinned artifact paths against the
# project root; the `three_head` package is imported from the experiment root.
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT
# Only two entries: this directory, for the self-contained `three_head` package,
# and the frozen upstream simulator. No sibling experiment folder is on the path.
for source in (
    EXPERIMENT_ROOT,
    PROJECT_ROOT / "memintelli_surrogate_comparison" / "upstream",
):
    sys.path.insert(0, str(source))

from three_head.core import collect_all, load_protocol, resolve_device  # noqa: E402
from three_head.evaluation import analyze  # noqa: E402
from three_head.training import train_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Consolidated three-head ideal-dynamic surrogate pipeline."
    )
    parser.add_argument("command", choices=("collect", "train", "analyze", "all"))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=EXPERIMENT_ROOT / "configs" / "three_head_protocol.json",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol, fingerprint = load_protocol(args.protocol.resolve(), PROJECT_ROOT)
    device = resolve_device(args.device)

    if args.command in ("collect", "all"):
        collect_all(protocol, fingerprint, PROJECT_ROOT, EXPERIMENT_ROOT, device)
    if args.command in ("train", "all"):
        summary = train_all(protocol, fingerprint, EXPERIMENT_ROOT, device)
        for head, record in summary["heads"].items():
            curve = record["curve"]
            print(
                f"[train] {head:<12}best={record['best_validation']:.6f}"
                f"@{record['best_step']} reports={curve['report_count']} "
                f"roughness={curve['late_curve_roughness']:.6f} "
                f"best_is_final={curve['best_is_final_step']}",
                flush=True,
            )
    if args.command in ("analyze", "all"):
        summary = analyze(protocol, fingerprint, EXPERIMENT_ROOT, device)
        for head, ratio in summary["conditional_to_exact_floor_ratio"].items():
            metrics = summary["head_metrics"][head]
            print(
                f"[test] {head:<12}floor={metrics['exact_independent_split']['mean']:.4f} "
                f"conditional={metrics['conditional']['mean']:.4f} ratio={ratio:.3f}",
                flush=True,
            )
        for method, spaces in summary["distribution_to_exact_floor_ratio"].items():
            if method == "nominal_only":
                continue
            print(
                f"[test] floor-ratio {method:<34}"
                f"var={spaces['linear_output']['variance_relative_l1']:.3f} "
                f"cov={spaces['linear_output']['covariance_relative_frobenius']:.3f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
