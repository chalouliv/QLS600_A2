"""Model SELECTION: which structure? Each candidate is fully fitted, then ranked
by held-out RMSE (last VALID_FRAC of the trace -- never used in fitting).

Exponent grid over gate powers:
  m^pm * h^ph [* n^pn],  pm in {1,2,3}, ph in {1,2}, pn in {1,2}
  (3 mh baselines + 12 with the slow n gate; no leak -- it never helped)
Plus: continuous-p discriminator (fit p as a real number; if it lands on ~1,
the data wants m^1).

Within each structure: batch nonlinear least squares (scipy least_squares),
multistarted; the restart with the best VALIDATION error is kept.

Also reports R^2, AIC, BIC per structure, plus an estimated measurement-noise
floor from quasi-constant-V windows: a valid RMSE sitting at the floor means
the model has explained everything explainable (it cannot go below the floor
without fitting noise).

Run:  python select_model.py [datafile] [n_restarts] [n_jobs] [n_trials]
  e.g.  python select_model.py data/CookLab1UnknownChannelExtra.txt 5 12 5
Fits are embarrassingly parallel: every (structure, restart) is an independent
job farmed out over n_jobs processes (default: all cores). The whole grid is
repeated n_trials times with different seeds; the table reports mean +/- std
of best-valid-RMSE per structure across trials. (Average metrics and
predictions across runs -- never average parameter vectors: the mean of two
local optima is generally not an optimum.)
Needs: numpy, scipy, hh_core.py
"""
import os
import sys
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
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


def _worker_fit(payload):
    """One (structure, restart) fit. Top-level so it pickles for multiprocessing.

    Returns dict with metrics + fitted q. Deterministic in (structure, restart)
    via the seed; independent of all other fits.
    """
    import numpy as np
    from scipy.optimize import least_squares
    from hh_core import simulate, unpack, q0_base, bounds
    (pm, ph, pn, uh, ul, un, j,
     Vtr, ttr, Itr, Vva, tva, Iva, seed) = payload
    LB, UB = bounds()
    n = len(Itr)
    k = len(q0_base()) - (0 if uh else 6) - (0 if ul else 2) - (0 if un else 6)
    rng = np.random.default_rng(seed)
    q0 = q0_base() if j == 0 else q0_base() + rng.normal(0, 0.35, len(q0_base()))

    def residuals(q):
        pred = simulate(Vtr, ttr, unpack(q), p=pm, ph=ph, pn=pn,
                        use_h=uh, use_leak=ul, use_n=un)
        return pred - Itr

    r = least_squares(residuals, q0, bounds=(LB, UB), method='trf',
                      ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=300)
    qh = unpack(r.x)
    pred_tr = simulate(Vtr, ttr, qh, p=pm, ph=ph, pn=pn,
                       use_h=uh, use_leak=ul, use_n=un)
    pred_va = simulate(Vva, tva, qh, p=pm, ph=ph, pn=pn,
                       use_h=uh, use_leak=ul, use_n=un)
    tr = float(np.sqrt(np.mean((pred_tr - Itr) ** 2)))
    va = float(np.sqrt(np.mean((pred_va - Iva) ** 2)))
    ss = lambda yt, yp: float(1 - np.sum((yt - yp) ** 2) /
                              np.sum((yt - np.mean(yt)) ** 2))
    r2tr, r2va = ss(Itr, pred_tr), ss(Iva, pred_va)
    rss = float(np.sum((pred_tr - Itr) ** 2))
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return dict(pm=pm, ph=ph, pn=pn, uh=uh, ul=ul, un=un, j=j,
                tr=tr, va=va, r2tr=r2tr, r2va=r2va, aic=aic, bic=bic, q=r.x)


def fit_structure(pm, ph, pn, use_h, use_leak, use_n=False, n_restarts=5, verbose=False):
    """Fit ONE structure (serial multistart; kept for API compatibility).

    Returns (train_rmse, valid_rmse, train_r2, valid_r2, aic, bic, q_hat)."""
    Vtr, ttr, Itr = V[train], t[train], I[train]
    Vva, tva, Iva = V[valid], t[valid], I[valid]
    best = None
    for j in range(n_restarts):
        d = _worker_fit((pm, ph, pn, use_h, use_leak, use_n, j,
                         Vtr, ttr, Itr, Vva, tva, Iva, 100000 + j))
        if verbose:
            print(f"    restart {j}: train={d['tr']:.5f} (R2={d['r2tr']:.4f})"
                  f" valid={d['va']:.5f} (R2={d['r2va']:.4f})")
        if best is None or d['va'] < best[1]:   # select restarts by VALID error
            best = (d['tr'], d['va'], d['r2tr'], d['r2va'],
                    d['aic'], d['bic'], d['q'])
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
    N_RESTARTS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    N_JOBS = int(sys.argv[3]) if len(sys.argv) > 3 else mp.cpu_count()
    N_TRIALS = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    print(f"data: {DATA}   n_train={train.sum()}  n_valid={valid.sum()}   "
          f"restarts={N_RESTARTS}  workers={N_JOBS}  trials={N_TRIALS}")
    nf = noise_floor(V, I)
    print(f"estimated measurement-noise SD (flat-V windows): {nf:.5f}")
    print("  -> no honest model can score below this on new data.\n")

    # exponent grid: m^pm * h^ph [* n^pn], no leak (leak never helped)
    STRUCTURES = []
    for pm in (1, 2, 3):
        STRUCTURES.append((pm, 1, 1, True, False, False))   # m^pm * h baselines
    for pm in (1, 2, 3):
        for ph in (1, 2):
            for pn in (1, 2):
                STRUCTURES.append((pm, ph, pn, True, False, True))  # with n

    # farm out every (structure, restart) independently, repeated N_TRIALS
    # times with different master seeds; aggregate best-valid per structure
    # across trials -> mean +/- std. Never average parameter vectors: the
    # mean of two local optima is generally not an optimum.
    Vtr, ttr, Itr = V[train], t[train], I[train]
    Vva, tva, Iva = V[valid], t[valid], I[valid]
    trial_rows = []
    for trial in range(N_TRIALS):
        jobs = []
        for si, (pm, ph, pn, uh, ul, un) in enumerate(STRUCTURES):
            for j in range(N_RESTARTS):
                jobs.append((pm, ph, pn, uh, ul, un, j,
                             Vtr, ttr, Itr, Vva, tva, Iva,
                             trial * 1000000 + si * 131 + j))
        print(f"trial {trial + 1}/{N_TRIALS}: dispatching {len(jobs)} fits...",
              flush=True)
        with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
            done = list(ex.map(_worker_fit, jobs))
        # best restart per structure (by valid RMSE) within this trial
        best = {}
        for d in done:
            key = (d['pm'], d['ph'], d['pn'], d['uh'], d['ul'], d['un'])
            if key not in best or d['va'] < best[key]['va']:
                best[key] = d
        rows_t = []
        for (pm, ph, pn, uh, ul, un) in STRUCTURES:
            d = best[(pm, ph, pn, uh, ul, un)]
            name = f"m^{pm} h^{ph}" + (f" n^{pn}" if un else "")
            rows_t.append(dict(name=name, pm=pm, ph=ph, pn=pn,
                               use_h=uh, use_leak=ul, use_n=un,
                               tr=d['tr'], va=d['va'], r2va=d['r2va'],
                               bic=d['bic'], q=list(d['q'])))
        trial_rows.append(rows_t)
        bv = min(rows_t, key=lambda r: r['va'])
        print(f"  trial {trial + 1} best: {bv['name']} valid={bv['va']:.5f}",
              flush=True)
    print("\nall trials done.\n", flush=True)

    # aggregate across trials
    agg = []
    for si in range(len(STRUCTURES)):
        vas = np.array([trial_rows[t][si]['va'] for t in range(N_TRIALS)])
        bics = np.array([trial_rows[t][si]['bic'] for t in range(N_TRIALS)])
        r2s = np.array([trial_rows[t][si]['r2va'] for t in range(N_TRIALS)])
        i_min = int(np.argmin(vas))
        agg.append(dict(name=trial_rows[0][si]['name'],
                        va_mean=float(np.mean(vas)), va_std=float(np.std(vas)),
                        va_min=float(np.min(vas)),
                        r2_mean=float(np.mean(r2s)),
                        bic_mean=float(np.mean(bics)),
                        q_min=list(trial_rows[i_min][si]['q'])))
    agg.sort(key=lambda r: r['va_mean'])
    print(f"{'structure':30s} {'valid_mean':>10s} {'valid_std':>9s} "
          f"{'valid_min':>9s} {'R2_mean':>8s} {'BIC_mean':>9s}")
    for r in agg:
        print(f"{r['name']:30s} {r['va_mean']:10.5f} {r['va_std']:9.5f} "
              f"{r['va_min']:9.5f} {r['r2_mean']:8.4f} {r['bic_mean']:9.1f}")
    by_valid = agg[0]
    by_bic = min(agg, key=lambda r: r['bic_mean'])
    print(f"\nWINNER by held-out RMSE (mean): {by_valid['name']}  "
          f"valid={by_valid['va_mean']:.5f} +/- {by_valid['va_std']:.5f}")
    print(f"WINNER by BIC (mean):           {by_bic['name']}  "
          f"BIC={by_bic['bic_mean']:.1f}")
    if by_valid['va_mean'] <= nf * 1.15:
        print("held-out RMSE is at the noise floor: nothing left to explain.")
    else:
        print(f"held-out RMSE is {by_valid['va_mean']/nf:.2f}x the noise floor: "
              f"structural gap remains (or noisier regime).")

    print("\n--- exponent discriminator: fit p as a free parameter ---")
    ps = fit_p_free()
    for j, p in enumerate(ps):
        print(f"  start {j}: p_hat = {p:.3f}")
    print(f"  median p_hat = {np.median(ps):.3f}  ->  "
          f"{'m^1 confirmed' if abs(np.median(ps) - 1) < 0.15 else 'inconclusive'}")

    import json
    out_path = os.path.join(_here, 'selection_results.json')
    with open(out_path, 'w') as f:
        json.dump(dict(data=DATA, noise_floor=nf, n_train=int(train.sum()),
                       n_valid=int(valid.sum()), n_trials=N_TRIALS,
                       aggregate=agg, trials=trial_rows,
                       winner_valid=by_valid['name'], winner_bic=by_bic['name'],
                       p_discriminator=[float(x) for x in ps]), f, indent=1)
    print(f"\nresults saved to {out_path}")
