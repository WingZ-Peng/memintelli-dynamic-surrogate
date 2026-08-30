# Ideal-Conductance Dynamic Three-Head Surrogate

Weight, ideal quantized/sliced conductance, layout, and hardware config
are fixed. Write variation and stuck faults are excluded. All test inputs
are held out.

## Head diagnostics

Correlation rows are scored on the physical support: the 1,300 of 9,900
off-diagonal entries that can be non-zero. The rest are exactly zero in
truth, so scoring them measures the reference's sampling noise.

| Metric | Exact floor | Conditional | Shuffled input features | Shared | Floor ratio |
|---|---:|---:|---:|---:|---:|
| Mean NRMSE | 0.0442 | 0.0335 | 0.1626 | 0.1164 | 0.759 |
| Variance L1 | 0.0494 | 0.0398 | 0.6357 | 0.4597 | 0.805 |
| Correlation Fro. (support) | 0.1963 | 0.1429 | 1.3824 | 0.9895 | 0.728 |

## Where the correlation lives

| Pair type | Pairs | RMS rho | Share of off-diagonal energy |
|---|---:|---:|---:|
| same k, shared row | 900 | 0.0760 | 7.4% |
| same k, shared output column | 400 | 0.3762 | 80.6% |
| same k, disjoint (exactly 0) | 3600 | 0.0313 | 5.0% |
| different k_block (exactly 0) | 5000 | 0.0312 | 6.9% |

## Correlation predictors

| Predictor | Full matrix | Off-diagonal | Support only |
|---|---:|---:|---:|
| `exact_independent_split` | 0.3352 | 0.5239 | 0.1963 |
| `identity` | 0.6409 | 1.0000 | 1.0000 |
| `shared_empirical` | 0.6354 | 0.9913 | 0.9895 |
| `shuffled_input_features` | 0.8593 | 1.3417 | 1.3824 |
| `analytic_first_order` | 0.2491 | 0.3895 | 0.1874 |
| `structured_conditional` | 0.2382 | 0.3723 | 0.1429 |

## End-to-end distribution

### s_tile

| Method | Mean NRMSE | Variance L1 | Covariance Fro. | Quantile NRMSE | Sliced W. |
|---|---:|---:|---:|---:|---:|
| `exact_independent_split` | 0.0442 | 0.0494 | 0.3039 | 0.0826 | 0.0564 |
| `conditional_three_head` | 0.0456 | 0.0537 | 0.3049 | 0.0863 | 0.0572 |
| `analytic_correlation_three_head` | 0.0455 | 0.0537 | 0.3079 | 0.0861 | 0.0571 |
| `shuffled_input_features` | 0.1663 | 0.6376 | 1.0278 | 0.5562 | 0.1774 |
| `shared_three_head` | 0.1217 | 0.4602 | 0.7487 | 0.4017 | 0.1321 |
| `nominal_only` | 0.1195 | 1.0000 | 1.0000 | 1.3454 | 0.8005 |

### linear_output

| Method | Mean NRMSE | Variance L1 | Covariance Fro. | Quantile NRMSE | Sliced W. |
|---|---:|---:|---:|---:|---:|
| `exact_independent_split` | 0.0428 | 0.0496 | 0.2510 | 0.0807 | 0.0559 |
| `conditional_three_head` | 0.0451 | 0.0515 | 0.2517 | 0.0845 | 0.0574 |
| `analytic_correlation_three_head` | 0.0453 | 0.0513 | 0.2545 | 0.0850 | 0.0573 |
| `shuffled_input_features` | 0.1686 | 0.4865 | 0.8802 | 0.4384 | 0.1834 |
| `shared_three_head` | 0.1229 | 0.3443 | 0.6279 | 0.3168 | 0.1343 |
| `nominal_only` | 0.1215 | 1.0000 | 1.0000 | 1.3455 | 0.8019 |

### Distance to the Exact finite-sample floor

| Method | s_tile covariance | linear_output variance | linear_output covariance |
|---|---:|---:|---:|
| `exact_independent_split` | 1.000 | 1.000 | 1.000 |
| `conditional_three_head` | 1.003 | 1.038 | 1.003 |
| `analytic_correlation_three_head` | 1.013 | 1.035 | 1.014 |
| `shuffled_input_features` | 3.382 | 9.806 | 3.508 |
| `shared_three_head` | 2.464 | 6.941 | 2.502 |

## What this does not show

- One fixed mathematical weight and ideal conductance tensor.
- Write variation and stuck faults are deliberately excluded.
- One fixed shape, tile layout, and hardware configuration.
- Synthetic Gaussian inputs, not real-network activations.
- Head schedules and the correlation regularizer were chosen on the validation split; the test split is read only here.
