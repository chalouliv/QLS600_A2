"""Model-selection metrics registry for HH structure comparison.

Split out of select_model.py: every metric is a small function registered with
metadata (direction, cost, description). Callers pick which metrics to run.

HONEST-SCORING CONTRACT (fixed 2026-10-12):
  Every prediction fed into these metrics MUST come from a CONTINUOUS
  simulation of the full trace from t=0, sliced to the train/validation
  windows afterwards (hh_core.honest_predict). Predictions from standalone
  segment simulation carry an initialization transient (gates reset
  mid-protocol: m off by 0.80 at the Extra validation start) that corrupts
  every validation metric -- valid_rmse inflated, valid_r2 deflated to 0.786
  instead of 0.915, and the cross-structure ranking inverted (the fake
  "winner" m^3h^1n^2 scored 0.917 standalone vs 0.837 honest). This module
  cannot detect the corruption; it trusts its inputs. Feed it honest
  predictions.

Usage:
    from model_metrics import evaluate, list_metrics

    vals = evaluate(fit, data)                                  # all 'fast' metrics
    vals = evaluate(fit, data, which='all')                     # everything
    vals = evaluate(fit, data, which='medium')                  # cost group
    vals = evaluate(fit, data, which=['bic', 'valid_rmse'])     # explicit list

`fit` dict keys (all optional; a metric needing a missing key yields NaN):
    pred_tr, pred_va : model predictions on train / validation (array-likes)
                       -- MUST be slices of a continuous full-trace simulation
    rss_tr           : sum of squared train residuals (float)
    k                : number of active (fitted) parameters (int)
    n_train          : number of train samples (int)
    restart_vas      : validation RMSE of every multistart restart (list)

`data` dict keys:
    Itr, Iva        : measured current, train / validation
    Vva, tva        : voltage & time on validation (window/segment metrics)
    noise_floor     : measurement-noise SD estimate (float)

Costs: 'fast' = pure numpy on predictions (microseconds);
       'medium' = needs V/t windows or restart lists (still cheap, no refits).
A metric that raises returns NaN rather than killing a grid run.
"""
import numpy as np

METRICS = {}


def register(name, direction="lower", cost="fast", description="", needs=()):
    """Decorator registering a metric.

    direction: 'lower' (lower is better) or 'higher' (higher is better).
    """
    def deco(fn):
        METRICS[name] = dict(fn=fn, direction=direction, cost=cost,
                             description=description, needs=tuple(needs))
        return fn
    return deco


def _safe(fn):
    def wrap(fit, data):
        try:
            v = fn(fit, data)
            return float(v) if v is not None else float("nan")
        except Exception:
            return float("nan")
    return wrap


def _asarray(x):
    return np.asarray(x, dtype=float)


def _r2(yt, yp):
    yt, yp = _asarray(yt), _asarray(yp)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ---------------------------------------------------------------- fit quality
@register("train_rmse", direction="lower", cost="fast",
          description="RMSE on the training segment (honest: scored from "
                      "continuous simulation)",
          needs=("pred_tr", "Itr"))
@_safe
def _train_rmse(fit, data):
    return np.sqrt(np.mean((_asarray(data["Itr"]) - _asarray(fit["pred_tr"])) ** 2))


@register("valid_rmse", direction="lower", cost="fast",
          description="RMSE on the held-out validation segment (honest: "
                      "continuous simulation from t=0, then sliced)",
          needs=("pred_va", "Iva"))
@_safe
def _valid_rmse(fit, data):
    return np.sqrt(np.mean((_asarray(data["Iva"]) - _asarray(fit["pred_va"])) ** 2))


@register("train_mae", direction="lower", cost="fast",
          description="Mean absolute error, train (robust to outliers)",
          needs=("pred_tr", "Itr"))
@_safe
def _train_mae(fit, data):
    return np.mean(np.abs(_asarray(data["Itr"]) - _asarray(fit["pred_tr"])))


@register("valid_mae", direction="lower", cost="fast",
          description="Mean absolute error, validation (robust to outliers)",
          needs=("pred_va", "Iva"))
@_safe
def _valid_mae(fit, data):
    return np.mean(np.abs(_asarray(data["Iva"]) - _asarray(fit["pred_va"])))


@register("train_r2", direction="higher", cost="fast",
          description="Fraction of variance explained, train",
          needs=("pred_tr", "Itr"))
@_safe
def _train_r2(fit, data):
    return _r2(data["Itr"], fit["pred_tr"])


@register("valid_r2", direction="higher", cost="fast",
          description="Fraction of variance explained, validation (honest)",
          needs=("pred_va", "Iva"))
@_safe
def _valid_r2(fit, data):
    return _r2(data["Iva"], fit["pred_va"])


@register("noise_ceiling_r2", direction="higher", cost="fast",
          description="Best R^2 any model could reach given the noise floor",
          needs=("Iva", "noise_floor"))
@_safe
def _noise_ceiling_r2(fit, data):
    var = np.var(_asarray(data["Iva"]))
    nf = float(data["noise_floor"])
    return 1.0 - nf ** 2 / var if var > 0 else float("nan")


@register("max_abs_err_valid", direction="lower", cost="fast",
          description="Worst single-sample |error| on validation (spike misses)",
          needs=("pred_va", "Iva"))
@_safe
def _max_abs_err_valid(fit, data):
    return np.max(np.abs(_asarray(data["Iva"]) - _asarray(fit["pred_va"])))


@register("rmse_over_noise_floor", direction="lower", cost="fast",
          description="valid_rmse / noise_floor; 1.0 = at the floor",
          needs=("pred_va", "Iva", "noise_floor"))
@_safe
def _rmse_over_noise_floor(fit, data):
    rmse = np.sqrt(np.mean((_asarray(data["Iva"]) - _asarray(fit["pred_va"])) ** 2))
    return rmse / float(data["noise_floor"])


@register("resid_std_over_floor", direction="lower", cost="fast",
          description="SD of validation residuals / noise floor; want ~1.0",
          needs=("pred_va", "Iva", "noise_floor"))
@_safe
def _resid_std_over_floor(fit, data):
    r = _asarray(data["Iva"]) - _asarray(fit["pred_va"])
    return np.std(r) / float(data["noise_floor"])


@register("resid_lag1_autocorr_valid", direction="lower", cost="fast",
          description="|lag-1 autocorrelation| of validation residuals; want ~0 (white)",
          needs=("pred_va", "Iva"))
@_safe
def _resid_lag1_autocorr_valid(fit, data):
    r = _asarray(data["Iva"]) - _asarray(fit["pred_va"])
    r = r - np.mean(r)
    denom = np.sum(r ** 2)
    if denom == 0 or len(r) < 3:
        return float("nan")
    return abs(np.sum(r[1:] * r[:-1]) / denom)


# ------------------------------------------------------- complexity/parsimony
@register("n_params", direction="lower", cost="fast",
          description="Number of active fitted parameters",
          needs=("k",))
@_safe
def _n_params(fit, data):
    return float(fit["k"])


def _ic(fit, data, penalty):
    rss = float(fit["rss_tr"])
    n = int(fit["n_train"])
    k = int(fit["k"])
    return n * np.log(rss / n) + penalty * k


@register("aic", direction="lower", cost="fast",
          description="Akaike information criterion (train RSS)",
          needs=("rss_tr", "k", "n_train"))
@_safe
def _aic(fit, data):
    return _ic(fit, data, 2.0)


@register("bic", direction="lower", cost="fast",
          description="Bayesian information criterion (train RSS)",
          needs=("rss_tr", "k", "n_train"))
@_safe
def _bic(fit, data):
    return _ic(fit, data, float(np.log(int(fit["n_train"]))))


@register("aicc", direction="lower", cost="fast",
          description="AIC with small-sample correction",
          needs=("rss_tr", "k", "n_train"))
@_safe
def _aicc(fit, data):
    n, k = int(fit["n_train"]), int(fit["k"])
    base = _ic(fit, data, 2.0)
    return base + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else float("nan")


# ------------------------------------------------- robustness / segment-wise
@register("worst_window_rmse_valid", direction="lower", cost="medium",
          description="Max RMSE over sliding 50-sample windows of validation",
          needs=("pred_va", "Iva"))
@_safe
def _worst_window_rmse_valid(fit, data):
    r = _asarray(data["Iva"]) - _asarray(fit["pred_va"])
    W, step = 50, 10
    worst = 0.0
    for i in range(0, len(r) - W + 1, step):
        w = np.sqrt(np.mean(r[i:i + W] ** 2))
        if w > worst:
            worst = w
    return worst


@register("steady_state_mae_valid", direction="lower", cost="medium",
          description="MAE inside flat-V windows of validation (steady-state I-V)",
          needs=("pred_va", "Iva", "Vva"))
@_safe
def _steady_state_mae_valid(fit, data):
    V = _asarray(data["Vva"])
    r = np.abs(_asarray(data["Iva"]) - _asarray(fit["pred_va"]))
    W, step = 20, 5
    flat = np.zeros(len(V), dtype=bool)
    for i in range(0, len(V) - W + 1, step):
        if V[i:i + W].max() - V[i:i + W].min() < 1.0:
            flat[i:i + W] = True
    return float(np.mean(r[flat])) if flat.any() else float("nan")


@register("peak_capture_rel_err_valid", direction="lower", cost="medium",
          description="|largest |I| excursion| relative error on validation",
          needs=("pred_va", "Iva"))
@_safe
def _peak_capture_rel_err_valid(fit, data):
    yt, yp = _asarray(data["Iva"]), _asarray(fit["pred_va"])
    denom = np.max(np.abs(yt))
    return abs(np.max(np.abs(yp)) - denom) / denom if denom > 0 else float("nan")


@register("multistart_spread_valid", direction="lower", cost="medium",
          description="SD of validation RMSE across multistart restarts (optimizer stability)",
          needs=("restart_vas",))
@_safe
def _multistart_spread_valid(fit, data):
    vas = [float(v) for v in fit["restart_vas"]]
    return float(np.std(vas)) if len(vas) >= 2 else float("nan")


# ------------------------------------------------------------------ interface
def _resolve_names(which):
    """Public: resolve a metric selection to an ordered name list."""
    if which is None:
        return [n for n, m in METRICS.items() if m["cost"] == "fast"]
    if isinstance(which, str):
        if which == "all":
            return list(METRICS)
        if which in ("fast", "medium"):
            return [n for n, m in METRICS.items() if m["cost"] == which]
        which = [which]
    names = list(which)
    unknown = [n for n in names if n not in METRICS]
    if unknown:
        raise KeyError(f"unknown metrics: {unknown}; see list_metrics()")
    return names


# backwards-compatible alias used inside evaluate()
resolve_names = _resolve_names


def evaluate(fit, data, which=None):
    """Compute selected metrics for one fitted structure.

    Returns dict {metric_name: value}. A metric whose inputs are missing
    (or that raises) yields NaN instead of aborting the run.

    CONTRACT: pred_tr / pred_va must be slices of a CONTINUOUS full-trace
    simulation (see hh_core.honest_predict). Standalone segment predictions
    corrupt every validation metric via the initialization transient.
    """
    out = {}
    for name in _resolve_names(which):
        out[name] = METRICS[name]["fn"](fit, data)
    return out


def list_metrics(cost=None):
    """Human-readable listing: name, direction, cost, description."""
    lines = []
    for name, m in METRICS.items():
        if cost is not None and m["cost"] != cost:
            continue
        arrow = "v" if m["direction"] == "lower" else "^"
        lines.append(f"  {name:28s} {arrow} [{m['cost']:6s}] {m['description']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"{len(METRICS)} metrics registered:\n")
    print(list_metrics())
    print("\n'higher is better' marked ^, 'lower is better' marked v.")
