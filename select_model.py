"""Model SELECTION: which structure? Each candidate is fully fitted, then ranked
by held-out RMSE (last 30% of the noise segment -- never used in fitting).

Grid over structures:
  p in {1,2,3}          -- gate exponent (m^p)
  use_h in {True,False} -- inactivation gate present or not
  use_leak in {True,False} -- leak conductance present or not
Plus: continuous-p discriminator (fit p as a real number; if it lands on ~1,
the data wants m^1).

Within each structure: batch nonlinear least squares (scipy least_squares),
multistarted; the restart with the best VALIDATION error is kept.

Run:  python select_model.py     (~2-4 min)
Needs: numpy, scipy, hh_core.py
"""
import numpy as np
from scipy.optimize import least_squares

from hh_core import simulate, unpack, q0_base, bounds

d = np.loadtxt('CookLab1UnknownChannel.txt')

t, V, I = d[:, 0], d[:, 1], d[:, 2]
train = (t < 275) | ((t >= 275) & (t < 485))
valid = t >= 485
LB, UB = bounds()


def rmse(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def fit_structure(p, use_h, use_leak, n_restarts=3, verbose=False):
    """Fit ONE structure. Returns (train_rmse, valid_rmse, q_hat)."""
    def residuals(q):
        pred = simulate(V[train], t[train], unpack(q),
                        p=p, use_h=use_h, use_leak=use_leak)
        return pred - I[train]

    rng = np.random.default_rng(0)
    best = None
    for j in range(n_restarts):
        q0 = q0_base() if j == 0 else q0_base() + rng.normal(0, 0.35, 16)
        r = least_squares(residuals, q0, bounds=(LB, UB), method='trf',
                          ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=300)
        tr = rmse(r.fun)
        va = rmse(simulate(V[valid], t[valid], unpack(r.x),
                           p=p, use_h=use_h, use_leak=use_leak) - I[valid])
        if verbose:
            print(f"    restart {j}: train={tr:.5f} valid={va:.5f}")
        if best is None or va < best[1]:   # select restarts by VALID error
            best = (tr, va, r.x)
    return best


def fit_p_free(n_restarts=5):
    """Discriminator: fit the exponent p as a continuous parameter.
    q = [16 model params, log(p)]. If p_hat -> ~1 from every start, m^1 wins."""
    LBp = np.append(LB, np.log(0.5))
    UBp = np.append(UB, np.log(4.0))

    def residuals(qp):
        p = float(np.exp(qp[16]))
        pred = simulate(V[train], t[train], unpack(qp[:16]),
                        p=p, use_h=True, use_leak=False)
        return pred - I[train]

    rng = np.random.default_rng(1)
    ps = []
    for j in range(n_restarts):
        q0 = np.append(q0_base(), np.log(2.0)) if j == 0 else \
             np.append(q0_base() + rng.normal(0, 0.35, 16), np.log(rng.uniform(0.7, 3.0)))
        r = least_squares(residuals, q0, bounds=(LBp, UBp), method='trf',
                          ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=300)
        ps.append(float(np.exp(r.x[16])))
    return ps


if __name__ == '__main__':
    STRUCTURES = [
        (1, True, False),    # m^1 * h, no leak  <- expected winner
        (2, True, False),
        (3, True, False),
        (1, False, False),   # no inactivation gate (ablation)
        (1, True, True),     # with leak (ablation)
        (2, True, True),
    ]
    print(f"{'structure':28s} {'train':>8s} {'valid':>8s}")
    rows = []
    for (p, uh, ul) in STRUCTURES:
        tr, va, _ = fit_structure(p, uh, ul)
        name = f"m^{p} * {'h' if uh else '-'} * {'leak' if ul else 'no leak'}"
        rows.append((va, name, tr))
        print(f"{name:28s} {tr:8.5f} {va:8.5f}")
    rows.sort()
    print(f"\nWINNER (lowest held-out RMSE): {rows[0][1]}  valid={rows[0][0]:.5f}")

    print("\n--- exponent discriminator: fit p as a free parameter ---")
    ps = fit_p_free()
    for j, p in enumerate(ps):
        print(f"  start {j}: p_hat = {p:.3f}")
    print(f"  median p_hat = {np.median(ps):.3f}  ->  "
          f"{'m^1 confirmed' if abs(np.median(ps) - 1) < 0.15 else 'inconclusive'}")
