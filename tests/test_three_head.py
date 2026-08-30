from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT
for source in (
    EXPERIMENT_ROOT,
    PROJECT_ROOT / "memintelli_surrogate_comparison" / "upstream",
):
    sys.path.insert(0, str(source))

from three_head.core import collect_all, load_protocol, resolve_device  # noqa: E402
from three_head.evaluation import analyze  # noqa: E402
from three_head.structure import (  # noqa: E402
    analytic_anchor,
    analytic_correlation,
    build_coordinate_structure,
    empirical_correlation,
)
from three_head.structured_correlation import StructuredSourceCorrelation  # noqa: E402
from three_head.training import curve_statistics, learning_rate_scale, train_all  # noqa: E402


class ThreeHeadPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol, cls.fingerprint = load_protocol(
            EXPERIMENT_ROOT / "configs" / "three_head_protocol.json", PROJECT_ROOT
        )
        cls.device = resolve_device("cuda:0")

    def test_package_is_self_contained(self) -> None:
        """Every import must resolve inside this package, torch, or memintelli.pimpy.

        The forbidden tokens are the module paths this package was extracted
        from; none of them exist in this repository, so any reappearance means a
        dependency on the original workspace has crept back in.
        """
        forbidden = ("from src.", "import src.", "memintelli_surrogate_compare",
                     "correlation_structured", "variance_stability", "ideal_dynamic.")
        for path in sorted((EXPERIMENT_ROOT / "three_head").glob("*.py")):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(
                    token, text, f"{path.name} still depends on {token!r}"
                )

    def test_learning_rate_schedule(self) -> None:
        steps, warmup = 2000, 50
        self.assertAlmostEqual(learning_rate_scale(50, steps, warmup, "cosine"), 1.0)
        self.assertAlmostEqual(
            learning_rate_scale(steps, steps, warmup, "cosine"), 0.0, places=12
        )
        self.assertEqual(learning_rate_scale(500, steps, 0, "constant"), 1.0)
        with self.assertRaises(ValueError):
            learning_rate_scale(1, steps, 0, "linear")

    def test_curve_statistics_flags_unconverged_runs(self) -> None:
        falling = [{"step": float(s), "validation": 0.5 - 0.1 * s} for s in (1, 2, 3, 4)]
        self.assertTrue(curve_statistics(falling, 4)["best_is_final_step"])
        flat = [{"step": float(s), "validation": 0.05} for s in (1, 2, 3, 4)]
        self.assertEqual(curve_statistics(flat, 4)["late_curve_roughness"], 0.0)

    def test_tiny_pipeline_and_identical_resume(self) -> None:
        tiny = copy.deepcopy(self.protocol)
        for name, count, seed, base in (
            ("train", 4, 701, 710_000_000),
            ("validation", 2, 702, 720_000_000),
            ("test", 2, 703, 730_000_000),
        ):
            tiny["splits"][name].update(
                {
                    "input_count": count,
                    "input_seed": seed,
                    "samples_per_context": 8,
                    "exact_seed_base": base,
                    "contexts_per_shard": 2,
                }
            )
        for head in ("mean", "variance", "correlation"):
            tiny["heads"][head].update(
                {
                    "hidden_dim": 16,
                    "coordinate_dim": 4,
                    "steps": 2,
                    "batch_contexts": 4,
                    "report_every": 1,
                    "warmup_steps": 1,
                }
            )
        tiny["evaluation"].update(
            {
                "exact_candidate_samples": 4,
                "exact_reference_samples": 4,
                "surrogate_samples": 4,
                "sliced_wasserstein_projections": 4,
            }
        )
        fingerprint = hashlib.sha256(
            json.dumps(tiny, sort_keys=True).encode("utf-8")
        ).hexdigest()

        with tempfile.TemporaryDirectory() as temporary:
            experiment_root = Path(temporary)
            collection = collect_all(
                tiny, fingerprint, PROJECT_ROOT, experiment_root, self.device
            )
            training = train_all(tiny, fingerprint, experiment_root, self.device)
            summary = analyze(tiny, fingerprint, experiment_root, self.device)
            resumed_collection = collect_all(
                tiny, fingerprint, PROJECT_ROOT, experiment_root, self.device
            )
            resumed_training = train_all(
                tiny, fingerprint, experiment_root, self.device
            )

            self.assertTrue(collection["complete"])
            self.assertTrue(training["complete"])
            self.assertTrue(summary["complete"])
            self.assertTrue(
                all(r["status"] == "reused" for r in resumed_collection["records"])
            )
            self.assertTrue(
                all(h["resumed_complete"] for h in resumed_training["heads"].values())
            )
            root = experiment_root / "outputs" / "three_head"
            self.assertTrue(root.exists())
            # The pinned v1 tree must never be written to.
            self.assertFalse((experiment_root / "outputs" / "checkpoints").exists())
            shard = torch.load(
                root / "context_shards" / "train" / "contexts_0000_0001.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(shard["mean_features"].shape[-1], 138)
            self.assertEqual(shard["variance_features"].shape[-1], 138)
            self.assertEqual(shard["correlation_features"].shape[-1], 118)
            self.assertEqual(shard["configuration"]["engine"]["write_variation"], 0.0)
            self.assertTrue(
                all(v for v in shard["checks"].values() if isinstance(v, bool))
            )

    def test_structured_correlation_is_exactly_zero_off_support(self) -> None:
        shard_directory = (
            EXPERIMENT_ROOT / "outputs" / "three_head" / "context_shards" / "test"
        )
        shards = sorted(shard_directory.glob("contexts_*.pt"))
        if not shards:
            self.skipTest("the full collect stage has not been run yet")
        shard = torch.load(shards[0], map_location="cpu", weights_only=False)
        structure = build_coordinate_structure(
            shard["valid_mask"], shard["configuration"]
        )
        declared = self.protocol["source_structure"]
        self.assertEqual(
            structure.voltage_group_count, int(declared["expected_voltage_groups"])
        )
        self.assertEqual(
            structure.conductance_group_count,
            int(declared["expected_conductance_groups"]),
        )
        self.assertEqual(
            int(structure.support_mask.sum()),
            int(declared["structurally_nonzero_offdiagonal_entries"]),
        )

        features = shard["correlation_features"].float()
        anchor = analytic_anchor(features)
        self.assertLess(float((anchor.square().sum(dim=-1) - 1.0).abs().max()), 1e-4)
        eye = torch.eye(structure.dim, dtype=torch.bool)
        off_support = ~(structure.support_mask | eye)

        analytic = analytic_correlation(anchor, structure)
        self.assertEqual(float(analytic[:, off_support].abs().max()), 0.0)

        torch.manual_seed(0)
        model = StructuredSourceCorrelation(
            feature_dim=features.shape[-1], structure=structure, source_dim=24
        )
        model.configure_normalization(features)
        model.eval()
        with torch.no_grad():
            correlation = model.correlation_matrix(features[:4])
        self.assertEqual(float(correlation[:, off_support].abs().max()), 0.0)
        diagonal = torch.diagonal(correlation, dim1=-2, dim2=-1)
        self.assertLess(float((diagonal - 1.0).abs().max()), 1e-5)
        self.assertGreater(
            float(torch.linalg.eigvalsh(correlation.double()).min()), 0.0
        )

    def test_offsupport_reference_is_pure_sampling_noise(self) -> None:
        shard_directory = (
            EXPERIMENT_ROOT / "outputs" / "three_head" / "context_shards" / "test"
        )
        shards = sorted(shard_directory.glob("contexts_*.pt"))
        if not shards:
            self.skipTest("the full collect stage has not been run yet")
        shard = torch.load(shards[0], map_location="cpu", weights_only=False)
        structure = build_coordinate_structure(
            shard["valid_mask"], shard["configuration"]
        )
        samples = shard["s_tile_valid_samples"].float()
        reference = empirical_correlation(samples)
        eye = torch.eye(structure.dim, dtype=torch.bool)
        off_support = ~(structure.support_mask | eye)
        observed = float(reference[:, off_support].pow(2).mean().sqrt())
        expected = (1.0 / samples.shape[1]) ** 0.5
        self.assertLess(abs(observed - expected), 0.2 * expected)


if __name__ == "__main__":
    unittest.main()
