# QLSC 600 Module 2 — Codebase Guide

For Jialin, Riv, Juliana. What each file does and how to use it.

## Files

| File | What it is |
|---|---|
| `CookLab1UnknownChannelExtra.txt` | The data. 3 columns: time (ms), voltage (mV), current. 2260 pts: holds at −60 mV, steps to −100…+20 mV, then random-voltage noise. |
| `hh_core.py` | The model. `simulate(V, t, params)` predicts current. Also `gating()` (Boltzmann + tau bells), `unpack()` (optimizer space → physical), `q0_base()`, `bounds()`. |
| `select_model.py` | The training pipeline. Fits 15 structures × 5 restarts in parallel, picks winners. Run: `python3 select_model.py 5 12 1` (restarts, workers, trials). |
| `model_metrics.py` | Metric registry (R², RMSE, BIC, AIC, MAE…). Used by `select_model.py`. |
| `run_vclamp_model.py` | **Submission file.** `run_vclamp_model(folder)` → predicted current array. Numpy only, locked m¹h¹ params. TA runs this on the hidden test file. |
| `selection_results.json` | Output of `select_model.py`. All scores, winners, fitted params. |
| `plot_actual_vs_sim.py` | Makes `actual_vs_sim.png` (validation overlay + parity plots). |

## The model (30 seconds)

`I = ḡ·m·h·(V − E_rev)` — one activation gate (m), one inactivation gate (h), no leak.

Each gate chases a voltage-dependent target: `dm/dt = (m∞(V) − m)/τ(V)`. Targets are Boltzmann sigmoids, speeds are bell curves. 14 fitted params total.

## How to read `selection_results.json`

- `aggregate`: one entry per structure, sorted by valid RMSE. Key fields: `name` (e.g. `"m^1 h^1"`), `va_mean` (valid RMSE), `r2_mean` (valid R²), `bic_mean`, `q_min` (fitted param vector, log-space).
- `winner_valid` / `winner_bic`: winners by each criterion. **BIC is the one we trust** (valid RMSE ties when the window is flat).
- `p_discriminator`: free-exponent fit; ≈1.0 confirms m¹. (0.5 = hit the lower bound, inconclusive.)
- `scoring`: should say `"honest-continuous-from-t0"` — full-trace simulation, then slice. Never score a mid-trace window standalone (gate reset bug).

## Structure tuples

`(pm, ph, pn, use_h, use_leak, use_n)` — e.g. `(1, 1, 1, True, False, False)` = m¹h¹.
`pm/ph/pn` = gate exponents; `use_*` = include that gate. Leak always off (fitted ≈ 0).

## Running things

```bash
# Full selection grid (~minutes on 12 cores)
python3 select_model.py 5 12 1

# Test the submission file locally (has a DELETE-BEFORE-SUBMISSION metrics block)
python3 run_vclamp_model.py

# Plot a trace
python3 plot_trace.py <file.txt> [out.png]
```

## Key numbers

- m¹h¹ locked: ḡ=0.001247, E_rev=−19.40 mV, Vm=−61.77, Vh=−2.11
- Train RMSE 0.00594, honest valid R² 0.915 (noise floor 0.00489)
- Thursday: TA runs `run_vclamp_model.py`, returns R². Guillaume presents over Zoom.

## Gotchas

- `q` (optimizer space) ≠ physical params. `unpack()` converts (log entries get `exp()`).
- Never average param vectors across restarts — take the winner's.
- The 2-point validation in the current JSON is a placeholder (lazy split); the real validation story is the 85/15 runs (R² 0.915).
