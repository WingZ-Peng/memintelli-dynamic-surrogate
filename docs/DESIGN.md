# Ideal-Conductance Dynamic Surrogate Contract

## Target

This experiment models one deterministic, ideally programmed conductance state:

```text
weight -> quantization/slicing -> G_ideal
G_ideal + dynamic read/voltage noise + ADC -> output distribution
```

Write variation and stuck faults are disabled. They are neither model inputs nor
part of the target distribution. The learned distribution is therefore
`p(Y | X, G_ideal, hardware_config)` and is not a device-population model.

## Head-specific inputs

All features are deterministic functions of the current input, `G_ideal`, the
fixed layout, and hardware configuration.

- Mean: quantized voltage, ideal conductance/level, ideal ADC state, ADC margin,
  and the analytic pre-ADC mean shift induced by log-normal read noise.
- Variance: squared voltage/conductance energy, analytic pre-ADC noise variance,
  ADC transition sensitivity, and local source-loading energy.
- Correlation: normalized voltage/read-noise source-loading signatures, ideal
  voltage/level geometry, ADC sensitivity, and coordinate/layout identity.

No head receives a realized `G_static`, a write seed, or a stochastic trace.

## Noise hierarchy

For a deterministic ideal cell conductance `g` and quantized voltage `v`:

```text
V_read = v * (1 + Normal(0, sigma_v))
G_read = g * exp(Normal(0, sigma_r))
pre_adc = sum(V_read * (G_read - LGS))
```

The feature builder uses the exact first two pre-ADC moments of this model.
Correlation source loadings use a local ADC transition gain, and the correlation
head is a factor model over those physical noise sources: it is initialized at
the closed-form analytic loading and learns a bounded correction plus a
per-coordinate explained-variance fraction.

## Scope boundary

This single pilot answers whether the three dynamic heads generalize to held-out
inputs for one fixed weight/configuration. It does not test write variation,
new weights, new shapes, new layouts, or new hardware configurations.
