"""Model SELECTION: which structure? Each candidate is fully fitted, then ranked
by held-out RMSE (last VALID_FRAC of the trace -- never used in fitting).

Grid over structures:
  p in {1,2,3}                -- gate exponent (m^p)
  use_h in {True,False}       -- inactivation gate present or not
  use_leak in {True,False}    -- leak conductance present or not
  use_n in {True,False}       -- slow second activation gate n or not
Plus: continuous-p discriminator (fit p as a real number; if it lands on ~1,
the data wants m^1).

Within each structure: batch nonlinear least squares (scipy least_squares),
multistarted; the restart with the best VALIDATION error is kept.

Also reports R^2, AIC, BIC per structure, plus an estimated measurement-noise
floor from quasi-constant-V windows: a valid RMSE sitting at the floor means
the model has explained everything explainable (it cannot go below the floor
without fitting noise).

Run:  python select_model.py [datafile]     (~2-4 min)
Needs: numpy, scipy, hh_core.py
"""
import os
import sys
import numpy as np
from scipy.optimize import least_squares

from hh_core import simulate, unpack, q0_base, bounds

_here = os.path.dirname(os.path.abspath(__file__))
_default = os.path.join(_here, 'data', 'CookLab1UnknownChannel.txt')
if not os.path.exists(_default):
    _default = 'CookLab1UnknownChannelExtra.txt'
DATA = sys.argv[1] if len(sys.argv) > 1 else _default
VALID_FRAC = 0.15  # last 15% of samples = held-out validation

d = np.loadtxt(DATA)
t, V, I = d[:, 0], d[:, 1], d[:, 2]
n_valid = int(len(t) * VALID_FRAC)
train = np.ones(len(t), dtype=bool)
train[-n_valid:] = False
valid = ~train
LB, UB = bounds()


def rmse(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def r2_score(y_true, y_pred):
    """1 - SS_res/SS_tot: fraction of variance explained."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')


def noise_floor(V, I, max_dV=1.0, W=20, step=5):
    """Median local std of I inside quasi-constant-V windows.

    Where V is (nearly) flat the true current is flat, so the local
    wobble is pure measurement noise. Its SD is the irreducible error
    floor: no model of I|V can score below it on new data without
    fitting noise.
    """
    sds = [I[i:i + W].std() for i in range(0, len(V) - W, step)
           if V[i:i + W].max() - V[i:i + W].min() < max_dV]
    return float(np.median(sds)) if sds else float('nan')


def n_active(use_h, use_leak, use_n=False):
    """Free parameters that actually touch the prediction."""
    k = len(q0_base())
    if not use_h:
        k -= 6    # Vh, kh, th_min, th_amp, th_c, th_w are dead
    if not use_leak:
        k -= 2    # g_L, E_L are dead
    if not use_n:
        k -= 6    # Vn, kn, tn_min, tn_amp, tn_c, tn_w are dead
    return k


def fit_structure(p, use_h, use_leak, use_n=False, n_restarts=3, verbose=False):
    """Fit ONE structure.
    Returns (train_rmse, valid_rmse, train_r2, valid_r2, aic, bic, q_hat)."""
    n = int(train.sum())
    k = n_active(use_h, use_leak, use_n)

    def residuals(q):
        pred = simulate(V[train], t[train], unpack(q),
                        p=p, use_h=use_h, use_leak=use_leak, use_n=use_n)
        return pred - I[train]

    rng = np.random.default_rng(0)
    best = None
    for j in range(n_restarts):
        q0 = q0_base() if j == 0 else q0_base() + rng.normal(0, 0.35, len(q0_base()))
        r = least_squares(residuals, q0, bounds=(LB, UB), method='trf',
                          ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=300)
        pred_tr = simulate(V[train], t[train], unpack(r.x),
                           p=p, use_h=use_h, use_leak=use_leak, use_n=use_n)
        pred_va = simulate(V[valid], t[valid], unpack(r.x),
                           p=p, use_h=use_h, use_leak=use_leak, use_n=use_n)
        tr = rmse(pred_tr - I[train])
        va = rmse(pred_va - I[valid])
        r2tr = r2_score(I[train], pred_tr)
        r2va = r2_score(I[valid], pred_va)
        rss = float(np.sum((pred_tr - I[train]) ** 2))
        aic = n * np.log(rss / n) + 2 * k
        bic = n * np.log(rss / n) + k * np.log(n)
        if verbose:
            print(f"    restart {j}: train={tr:.5f} (R2={r2tr:.4f})"
                  f" valid={va:.5f} (R2={r2va:.4f}) AIC={aic:.1f} BIC={bic:.1f}")
        if best is None or va < best[1]:   # select restarts by VALID error
            best = (tr, va, r2tr, r2va, aic, bic, r.x)
    return best


def fit_p_free(n_restarts=5):
    """Discriminator: fit the exponent p as a continuous parameter.
    q = [model params, log(p)]. If p_hat -> ~1 from every start, m^1 wins."""
    LBp = np.append(LB, np.log(0.5))
    UBp = np.append(UB, np.log(4.0))
    nq = len(q0_base())

    def residuals(qp):
        p = float(np.exp(qp[nq]))
        pred = simulate(V[train], t[train], unpack(qp[:nq]),
                        p=p, use_h=True, use_leak=False)
        return pred - I[train]

    rng = np.random.default_rng(1)
    ps = []
    for j in range(n_restarts):
        q0 = np.append(q0_base(), np.log(2.0)) if j == 0 else \
             np.append(q0_base() + rng.normal(0, 0.35, len(q0_base())), np.log(rng.uniform(0.7, 3.0)))
        r = least_squares(residuals, q0, bounds=(LBp, UBp), method='trf',
                          ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=300)
        ps.append(float(np.exp(r.x[nq])))
    return ps


if __name__ == '__main__':
    print(f"data: {DATA}   n_train={train.sum()}  n_valid={valid.sum()}")
    nf = noise_floor(V, I)
    print(f"estimated measurement-noise SD (flat-V windows): {nf:.5f}")
    print("  -> no honest model can score below this on new data.\n")

    STRUCTURES = [
        (1, True, False, False),   # m^1 * h, no leak  <- expected winner
        (2, True, False, False),
        (3, True, False, False),
        (1, False, False, False),  # no inactivation gate (ablation)
        (1, True, True, False),    # with leak (ablation)
        (2, True, True, False),
        (1, True, False, True),    # m * h * n (slow 2nd activation)
        (1, True, True, True),     # m * h * n + leak
    ]
    print(f"{'structure':30s} {'train':>8s} {'R2_tr':>7s} "
          f"{'valid':>8s} {'R2_va':>7s} {'AIC':>9s} {'BIC':>9s}")
    rows = []
    for (p, uh, ul, un) in STRUCTURES:
        tr, va, r2tr, r2va, aic, bic, _ = fit_structure(p, uh, ul, un)
        name = (f"m^{p} * {'h' if uh else '-'} * {'n' if un else '-'} * "
                f"{'leak' if ul else 'no leak'}")
        rows.append(dict(name=name, tr=tr, va=va, r2tr=r2tr, r2va=r2va,
                         aic=aic, bic=bic))
        print(f"{name:30s} {tr:8.5f} {r2tr:7.4f} {va:8.5f} "
              f"{r2va:7.4f} {aic:9.1f} {bic:9.1f}")
    by_valid = min(rows, key=lambda r: r['va'])
    by_bic = min(rows, key=lambda r: r['bic'])
    print(f"\nWINNER by held-out RMSE: {by_valid['name']}  "
          f"valid={by_valid['va']:.5f}  R2_va={by_valid['r2va']:.4f}")
    print(f"WINNER by BIC:           {by_bic['name']}  BIC={by_bic['bic']:.1f}")
    if by_valid['va'] <= nf * 1.15:
        print("held-out RMSE is at the noise floor: nothing left to explain.")
    else:
        print(f"held-out RMSE is {by_valid['va']/nf:.2f}x the noise floor: "
              f"structural gap remains (or noisier regime).")

    print("\n--- exponent discriminator: fit p as a free parameter ---")
    ps = fit_p_free()
    for j, p in enumerate(ps):
        print(f"  start {j}: p_hat = {p:.3f}")
    print(f"  median p_hat = {np.median(ps):.3f}  ->  "
          f"{'m^1 confirmed' if abs(np.median(ps) - 1) < 0.15 else 'inconclusive'}")
