#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QLSC 600 Module 2 — Hodgkin-Huxley unknown-channel model.

Model: I = g_bar * m * h * (V - E_rev), one activation (m) + one inactivation (h) gate.
Structure selected by 15-structure grid (BIC + honest held-out validation on
CookLab1UnknownChannelExtra.txt): m^1 h^1, no leak.
Fit: multistart nonlinear least squares, train RMSE 0.00594, honest validation R^2 0.9147.
"""

# %% SECTION 1: Packages import
import os
import numpy as np
import json as _json

# %% SECTION 2: Helper functions — parameters loaded from fit results JSON
import json as _json

_RESULTS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "selection_results.json")  # change per split
_STRUCTURE = "m^1 h^1"                                     # change as needed

def _load_params(json_path, structure):
    with open(json_path) as f:
        res = _json.load(f)
    row = next(r for r in res["aggregate"] if r["name"] == structure)
    q = row["q_min"]  # optimizer (log) space; unpack below
    return dict(
        g_bar=float(np.exp(q[0])),  e_rev=float(q[1]),
        vm=float(q[4]),  km=float(np.exp(q[5])),
        vh=float(q[6]),  kh=float(np.exp(q[7])),
        tm_min=float(np.exp(q[8])),  tm_amp=float(np.exp(q[9])),
        tm_c=float(q[10]),           tm_w=float(np.exp(q[11])),
        th_min=float(np.exp(q[12])), th_amp=float(np.exp(q[13])),
        th_c=float(q[14]),           th_w=float(np.exp(q[15])),
    )


_P = _load_params(_RESULTS_JSON, _STRUCTURE)
_G_BAR = _P["g_bar"]; _E_REV = _P["e_rev"]
_VM = _P["vm"]; _KM = _P["km"]; _VH = _P["vh"]; _KH = _P["kh"]
_TM_MIN = _P["tm_min"]; _TM_AMP = _P["tm_amp"]; _TM_C = _P["tm_c"]; _TM_W = _P["tm_w"]
_TH_MIN = _P["th_min"]; _TH_AMP = _P["th_amp"]; _TH_C = _P["th_c"]; _TH_W = _P["th_w"]


def _simulate_current(v, dt):
    """Predict current for a voltage-clamp waveform (mV), dt in ms."""
    v = np.asarray(v, dtype=float)

    def m_inf(x):
        return 1.0 / (1.0 + np.exp(-(x - _VM) / _KM))

    def h_inf(x):
        return 1.0 / (1.0 + np.exp((x - _VH) / _KH))

    def tau_m(x):
        return _TM_MIN + _TM_AMP * np.exp(-((x - _TM_C) / _TM_W) ** 2)

    def tau_h(x):
        return _TH_MIN + _TH_AMP * np.exp(-((x - _TH_C) / _TH_W) ** 2)

    # 100 ms pre-equilibration hold at v[0]: gates settle to their true
    # history-dependent state before the protocol starts, so the initial
    # transient cannot corrupt the prediction.
    n_hold = int(round(100.0 / dt))
    vv = np.concatenate([np.full(n_hold, v[0]), v])

    mi, tm = m_inf(vv), tau_m(vv)
    hi, th = h_inf(vv), tau_h(vv)
    am = np.exp(-dt / tm)
    bm = mi * (1.0 - am)
    ah = np.exp(-dt / th)
    bh = hi * (1.0 - ah)

    m = mi[0]
    h = hi[0]
    out = np.empty(len(vv))
    for k in range(len(vv)):
        m = am[k] * m + bm[k]
        h = ah[k] * h + bh[k]
        out[k] = _G_BAR * m * h * (vv[k] - _E_REV)
    return out[n_hold:]


# %% SECTION 3: Main function
def run_vclamp_model(folder: str, data: str = "CookLab1UnknownChannelExtra.txt"):
    """
    Compute current from voltage-clamp input.

    Parameters
    ----------
    folder :
        Directory with input file for voltage-clamp model.
    data :
        Name of input file.

    Returns
    -------
    np.ndarray
        Output current computed by the voltage-clamp model.
    """
    # %% SECTION 3-a: Load provided input data.
    # NO MODIFICATIONS: This section should be kept as is.
    data = np.loadtxt(os.path.join(folder, "CookLab1UnknownChannelExtra.txt"))
    v = data[:, 1]

    # i has the same shape as array v, add code in the next section to fill it
    i = np.empty_like(v)

    # %% SECTION 3-b: Compute i [filled by students]
    t = data[:, 0]
    dt = float(np.median(np.diff(t)))
    i = _simulate_current(v, dt)

    # %% SECTION 3-c: Assert validity of output and return it.
    # NO MODIFICATIONS: This section should be kept as is.
    assert len(i.shape) == 1, "Currrent array should be 1-dimensional"
    return i


# %% SECTION 4: Function calls
if __name__ == "__main__":
    # Replace this with the path to the folder where provided input .txt file is!
    folder = ""

    # Run function on the voltage-clamp data
    i = run_vclamp_model(folder)

    # ===== DELETE BEFORE SUBMISSION (local metrics check only) =====
    _d = np.loadtxt(os.path.join(folder, "CookLab1UnknownChannelExtra.txt"))
    _true = _d[:, 2]
    _r = _true - i
    _rmse = float(np.sqrt(np.mean(_r ** 2)))
    _r2 = float(1 - np.sum(_r ** 2) / np.sum((_true - _true.mean()) ** 2))
    print(f"RMSE: {_rmse:.5f} | R2: {_r2:.4f}")
    # ===== END DELETE BEFORE SUBMISSION =====
