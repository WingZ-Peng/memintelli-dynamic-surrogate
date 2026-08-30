**English** · [简体中文](README.zh-CN.md)

# Ideal-Conductance Dynamic Three-Head Surrogate

A learned distributional surrogate for a memristive crossbar matrix-vector
multiply. For **one fixed weight and one fixed ideal conductance tensor**, three
conditional heads predict the full output distribution that dynamic read and
voltage noise produces on held-out inputs — not just the mean, but the variance
and the correlation structure across output coordinates.

The learned object is `p(Y | X, G_ideal, hardware_config)`. Write variation and
stuck faults are switched off: they are neither inputs nor part of the target,
so this is deliberately **not** a device-population model.

## Result

On 64 held-out test contexts, the surrogate reproduces the simulator's output
covariance as closely as an independent draw from the simulator itself does:

| Method | Linear-output variance | Linear-output covariance |
|---|---:|---:|
| `exact_independent_split` (the reference) | 1.000 | 1.000 |
| **`conditional_three_head`** | **1.038** | **1.003** |
| `analytic_correlation_three_head` | 1.035 | 1.014 |
| `shared_three_head` (no conditioning) | 6.941 | 2.502 |
| `shuffled_input_features` (conditioning destroyed) | 9.806 | 3.508 |

These are ratios against an **Exact-vs-Exact finite-sample floor**: the 2,048
Exact samples per test context are split into 1,024 candidate and 1,024
reference halves, and the reference number is the distance from one half to the
other. A ratio of 1.000 means the surrogate is as close to the reference as a
second independent simulator run of the same size would be.

Per-head diagnostics, scored on held-out test contexts:

| Head | Exact floor | Conditional | Shared | Floor ratio |
|---|---:|---:|---:|---:|
| Mean NRMSE | 0.0442 | **0.0335** | 0.1164 | 0.759 |
| Variance L1 | 0.0494 | **0.0398** | 0.4597 | 0.805 |
| Correlation Frobenius (support) | 0.1963 | **0.1429** | 0.9895 | 0.728 |

`exact_independent_split` is the distance between *two noisy draws*, so it is
not a lower bound — a good deterministic predictor can legitimately score below
it. A ratio under 1.0 is not an impossibility, and one above 1.0 is not by
itself evidence of a bad head.

## Requirements

- Python 3.11
- PyTorch with a CUDA build matching your GPU (CPU works but is slow)
- `numpy < 2` and `matplotlib`, both pulled in by the vendored simulator
- About 300 MB of free disk for the regenerated sample shards

The published result was produced with Python 3.11.15 and torch 2.11.0+cu128 on
a single NVIDIA RTX 5060 (compute capability 12.0).

## 1. Set up the environment

```powershell
conda create -n surrogate python=3.11 -y
conda activate surrogate

# Install torch first, matching your CUDA version. This is the cu128 build used
# for the published result; pick your own from https://pytorch.org.
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

On Linux or macOS the same three commands work; only the activation syntax
differs (`conda activate` is identical, `pip` is identical).

## 2. Verify the installation

```powershell
python -m unittest discover -s tests -v
```

Expect **6 tests, 2 skipped** in a couple of seconds. The two skips are
expected on a fresh clone: they inspect collected sample shards, which are not
tracked in git (see step 4). The four that run include a complete tiny
end-to-end pipeline — collect, train, analyze, then resume — in a temporary
directory, so passing it means the whole stack is working.

## 3. Inspect the published result without recomputing

The trained checkpoints and the generated report ship with this repository, so
the result is readable immediately:

- [`docs/SUMMARY.md`](docs/SUMMARY.md) — the consolidated write-up: architecture,
  methodology, every table, and the verbatim execution log
- [`outputs/three_head/report/three_head_results.md`](outputs/three_head/report/three_head_results.md)
  — the generated report
- `outputs/three_head/report/three_head_results.json` — the same numbers, machine-readable
- `outputs/three_head/checkpoints/` — the three trained heads

To confirm the checkpoints match the protocol they were trained under:

```powershell
python .\scripts\write_summary.py
```

This regenerates `docs/SUMMARY.md` from the artifacts. It fails loudly if any
fingerprint disagrees.

## 4. Reproduce the result from scratch

```powershell
python .\scripts\run_three_head.py all --device cuda:0
```

Budget about **30 minutes** on an RTX 5060. Collection dominates at ~28 minutes
across 24 shards; training all three heads and running the analysis together
take under a minute.

The stages are separate and each is resumable, so you can run them one at a
time and interrupt freely:

```powershell
python .\scripts\run_three_head.py collect --device cuda:0   # ~28 min, writes 254 MB
python .\scripts\run_three_head.py train   --device cuda:0   # seconds
python .\scripts\run_three_head.py analyze --device cuda:0   # seconds
```

`--device` also accepts `auto` and `cpu`.

Re-running a completed stage does not recompute it. `collect` verifies each
existing shard and reports `reused`; `train` short-circuits to the stored best
model and reports `resumed_complete`. Resume is *identical*, not approximate —
optimizer and RNG generator state are checkpointed alongside the weights.

**Note on step 4 vs. step 3.** Running `analyze` overwrites
`outputs/three_head/report/`. The regenerated report reproduces the committed
one bit-for-bit when the protocol is unchanged, but if you have edited those
files by hand, back them up first.

## How it is built

`configs/three_head_protocol.json` is the single source of truth for every
count, seed, dimension and hyperparameter. Its raw bytes are hashed into a
`protocol_fingerprint` that is stamped into every shard, checkpoint and report.
Any artifact whose fingerprint disagrees is a hard error — there is no
permissive fallback and no "recompute if stale" path. **Editing the protocol
invalidates everything under `outputs/`.**

Features are deterministic functions of the input, `G_ideal`, the tile layout
and the hardware config — never of a realized `G_static`, a write seed, or a
stochastic trace. The pipeline computes the exact first two pre-ADC moments of
`V_read = v·(1+N(0,σ_v))` and `G_read = g·exp(N(0,σ_r))` analytically, converts
them into ADC code space, and estimates a local ADC transition gain.

### The correlation head is the substantive part

An `S_tile` coordinate is `(k_block, row_index, out_index)`. It reads voltages
on its own row within its own `k_block`, and conductance cells in its own output
column within its own `k_block`. Two coordinates are therefore functions of
**disjoint** random variables unless they share a `k_block` *and* either a row or
an output column — for every other pair the covariance is exactly zero, by
independence rather than by linearization.

Only 1,300 of the 9,900 off-diagonal entries can be non-zero. So the head is not
a dense low-rank matrix; it is a factor model over the physical noise sources,
with each coordinate written as a unit-variance sum:

```text
S_d = <alpha_d, eps_voltage[k, row]> + <beta_d, eps_conductance[k, out]>
      + sqrt(1 - |alpha_d|^2 - |beta_d|^2) * eta_d
```

Coordinates sharing no source are exactly uncorrelated, the matrix is positive
semi-definite for any parameter value, the diagonal is exactly one, and sampling
is native with no Cholesky factorization — all by construction rather than by
fitting. The closed-form loading is stored in the features, so training *starts*
at the analytic solution and is penalized for leaving it.

The `analytic_first_order` row in the report is that closed form with zero
parameters and no training. It is the ablation separating how much of the result
is physics from how much is learning.

## Repository layout

```text
three_head/                        the pipeline (self-contained)
  core.py                          collection, protocol, atomic writes
  features.py                      analytic pre-ADC moment features
  structure.py                     coordinate sharing + analytic covariance
  structured_correlation.py        the source factor model
  training.py                      all three heads, scheduled
  evaluation.py                    metrics, ablations, report
  metrics.py                       shared metric definitions
  conditional_mean.py   \
  observable_dpe.py      >          vendored model and tail code
  tail.py               /
configs/three_head_protocol.json   the single source of truth
scripts/run_three_head.py          the entry point
scripts/write_summary.py           regenerates docs/SUMMARY.md
tests/test_three_head.py           self-containment + pipeline + structure
outputs/three_head/                everything the pipeline writes
memintelli_surrogate_comparison/   the vendored upstream simulator, and the
                                   three files the protocol pins by hash
docs/                              the write-ups
```

That last directory name is not decorative: `configs/three_head_protocol.json`
records the SHA-256 of three files under
`memintelli_surrogate_comparison/artifacts/` and resolves them by that exact
relative path at startup. Renaming the directory would change the protocol and
invalidate the shipped checkpoints, so it is kept as-is.

### What ships, and what is regenerated

Tracked here: all source, the protocol, the three trained checkpoints (5 MB),
the generated reports, the collection and training manifests, and the execution
log. Not tracked: `outputs/three_head/context_shards/`, about 254 MB of
collected Exact samples, which `collect` regenerates deterministically from the
protocol seeds.

## Documentation

| Document | What it covers |
|---|---|
| [`docs/SUMMARY.md`](docs/SUMMARY.md) | Consolidated write-up, generated from the artifacts. Do not edit by hand — re-run `scripts/write_summary.py`. |
| [`docs/DESIGN.md`](docs/DESIGN.md) | The scope contract: what is fixed, what varies, what is excluded. |
| [`docs/KEY_FINDINGS.md`](docs/KEY_FINDINGS.md) | The findings in detail, including the negative results. |
| [`docs/PROGRESS_REPORT.md`](docs/PROGRESS_REPORT.md) | Full development history and the reasoning behind each design decision. |

Simplified Chinese translations exist for this README
([`README.zh-CN.md`](README.zh-CN.md)) and the scope contract
([`docs/DESIGN.zh-CN.md`](docs/DESIGN.zh-CN.md)). **The English versions are
authoritative for all numbers.** The other documents are English only:
`SUMMARY.md` is generated by `scripts/write_summary.py` and a translation of it
would go stale on the next run, and the two long-form reports are living
documents.

## Scope limits

This is a pilot. It shows a learnable, well-calibrated conditional distribution
under tightly controlled conditions, and nothing beyond them:

- One fixed mathematical weight and one fixed ideal conductance tensor.
- Write variation and stuck faults are deliberately excluded.
- One fixed shape, tile layout, and hardware configuration.
- Synthetic Gaussian inputs, not real-network activations.
- Head schedules and the correlation regularizer were chosen on the validation
  split. The test split is read exactly once, in `analyze`.
- **No speedup is claimed.** No wall-clock comparison against the simulator has
  been run.

## Attribution

The device simulator under `memintelli_surrogate_comparison/upstream/` is an
unmodified third-party snapshot of **MemIntelli**, from
<https://github.com/HUST-ISMD-Odyssey/Memintelli>, developed by Prof. Xiangshui
Miao and Prof. Yi Li's group at the Institute of Information Storage Materials
and Devices, Huazhong University of Science and Technology. If you use it,
cite their paper:

> H. Zhou, L. Yang, et al. *MemIntelli: A Generic End-to-End Simulation
> Framework for Memristive Intelligent Systems.* arXiv:2511.17418.

Only `memintelli.pimpy` is on this pipeline's import path; the rest of the
package is retained unmodified so it imports exactly as upstream published it.

**On its licensing.** The snapshot ships an MIT license at
`memintelli_surrogate_comparison/upstream/license.txt`, but the upstream README
also states that the model "is made publicly available on a non-commercial
basis." Those two statements are not consistent with each other, and this
repository does not attempt to resolve them. Nothing here grants you rights to
the upstream code beyond whatever its authors actually intend; check with them
before any use that depends on the answer.

Everything under `three_head/`, `configs/`, `scripts/`, `tests/` and `docs/` is
original work.
