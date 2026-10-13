"""Core HH model: shared by fitting scripts and the deliverable hh_model.py.

Model: I = g_bar * m^pm * h^ph * n^pn * (V - E) + g_L * (V - E_L)   (n optional)
  m_inf(V) = 1/(1+exp(-(V-Vm)/km))        (fast activation)
  h_inf(V) = 1/(1+exp(+(V-Vh)/kh))        (inactivation)
  n_inf(V) = 1/(1+exp(-(V-Vn)/kn))        (slow activation, optional)
  tau_X(V) = tX_min + tX_amp * exp(-((V-tX_c)/tX_w)^2)   (bell-shaped)
State update is the exact solution over each sample interval (piecewise-constant V):
  X <- X + (Xinf(V) - X) * (1 - exp(-dt/tau(V)))

HONEST-SCORING CONTRACT (fixed 2026-10-12):
  simulate() initializes gates at steady state of V[0]. That is correct ONLY
  when the trace starts at an equilibrated hold. NEVER score a mid-protocol
  segment by simulating it standalone -- the gates get reset to wrong values
  (measured: m off by 0.80, h off by 0.55 at the Extra validation start,
  predicted conductance off 8x, R^2 0.915 -> 0.786). Instead:
    * preferred: simulate the FULL trace from t=0, then slice the segment
      (see honest_predict() below), or
    * pass the true gate states as m0/h0/n0 if you have them.
"""
import numpy as np

PNAMES = ['g_bar', 'E', 'g_L', 'E_L', 'Vm', 'km', 'Vh', 'kh',
          'tm_min', 'tm_amp', 'tm_c', 'tm_w', 'th_min', 'th_amp', 'th_c', 'th_w',
          'Vn', 'kn', 'tn_min', 'tn_amp', 'tn_c', 'tn_w']
N_PARAMS = len(PNAMES)  # 22
DT = 0.25  # ms, sampling of the data file


def unpack(q):
    q = np.asarray(q, dtype=float)
    g_bar = np.exp(q[0]); E = q[1]; g_L = np.exp(q[2]); E_L = q[3]
    Vm = q[4]; km = np.exp(q[5]); Vh = q[6]; kh = np.exp(q[7])
    tm_min = np.exp(q[8]); tm_amp = np.exp(q[9]); tm_c = q[10]; tm_w = np.exp(q[11])
    th_min = np.exp(q[12]); th_amp = np.exp(q[13]); th_c = q[14]; th_w = np.exp(q[15])
    Vn = q[16]; kn = np.exp(q[17])
    tn_min = np.exp(q[18]); tn_amp = np.exp(q[19]); tn_c = q[20]; tn_w = np.exp(q[21])
    return (g_bar, E, g_L, E_L, Vm, km, Vh, kh,
            tm_min, tm_amp, tm_c, tm_w, th_min, th_amp, th_c, th_w,
            Vn, kn, tn_min, tn_amp, tn_c, tn_w)


def pack(params):
    (g_bar, E, g_L, E_L, Vm, km, Vh, kh,
     tm_min, tm_amp, tm_c, tm_w, th_min, th_amp, th_c, th_w,
     Vn, kn, tn_min, tn_amp, tn_c, tn_w) = params
    return np.array([np.log(g_bar), E, np.log(g_L), E_L, Vm, np.log(km), Vh, np.log(kh),
                     np.log(tm_min), np.log(tm_amp), tm_c, np.log(tm_w),
                     np.log(th_min), np.log(th_amp), th_c, np.log(th_w),
                     Vn, np.log(kn), np.log(tn_min), np.log(tn_amp), tn_c, np.log(tn_w)])


def gating(V, params):
    """Return (m_inf, tau_m, h_inf, tau_h) arrays for voltage array V (mV).

    Accepts the full 22-param vector (uses the first 16) or a dict."""
    if isinstance(params, dict):
        params = [params[k] for k in PNAMES]
    (g_bar, E, g_L, E_L, Vm, km, Vh, kh,
     tm_min, tm_amp, tm_c, tm_w, th_min, th_amp, th_c, th_w) = params[:16]
    V = np.asarray(V, dtype=float)
    m_inf = 1.0 / (1.0 + np.exp(-(V - Vm) / km))
    tau_m = tm_min + tm_amp * np.exp(-((V - tm_c) / tm_w) ** 2)
    h_inf = 1.0 / (1.0 + np.exp((V - Vh) / kh))
    tau_h = th_min + th_amp * np.exp(-((V - th_c) / th_w) ** 2)
    return m_inf, tau_m, h_inf, tau_h


def gating_n(V, params):
    """Second activation gate n (slower): (n_inf, tau_n) for voltage array V.

    Same functional form as m (increasing logistic + bell tau), own parameters
    params[16:22] = (Vn, kn, tn_min, tn_amp, tn_c, tn_w). Accepts the same
    dict-or-sequence params as gating()."""
    if isinstance(params, dict):
        params = [params[k] for k in PNAMES]
    Vn, kn, tn_min, tn_amp, tn_c, tn_w = params[16], params[17], params[18], params[19], params[20], params[21]
    V = np.asarray(V, dtype=float)
    n_inf = 1.0 / (1.0 + np.exp(-(V - Vn) / kn))
    tau_n = tn_min + tn_amp * np.exp(-((V - tn_c) / tn_w) ** 2)
    return n_inf, tau_n


def simulate(V_mV, t_ms, params, p=1, ph=1, pn=1, use_h=True, use_leak=True,
             use_n=False, dt=None, m0=None, h0=None, n0=None):
    """Predict current for a voltage-clamp waveform.

    V_mV, t_ms: array-likes of equal length (t in ms, uniformly sampled).
    params: dict or sequence of the 22 physical parameters (see PNAMES).
    p, ph, pn: exponents on the m, h, n gates (p kept as alias for the m exponent).
    use_n: include the slow second activation gate (I *= n^pn).
    m0, h0, n0: optional explicit initial gate states ("better starting values").
        If None (default), gates start at steady state of V_mV[0] -- correct
        only when the trace starts at an equilibrated hold. For a segment
        starting mid-protocol, either simulate the full trace from t=0 and
        slice (preferred; see honest_predict), or pass the true states here.
    Returns: predicted current array (same units as fitted data).
    """
    if isinstance(params, dict):
        params = [params[k] for k in PNAMES]
    V = np.asarray(V_mV, dtype=float)
    t = np.asarray(t_ms, dtype=float)
    if dt is None:
        dt = float(np.median(np.diff(t)))
    g_bar, E, g_L, E_L = params[0], params[1], params[2], params[3]
    m_inf, tau_m, h_inf, tau_h = gating(V, params)
    am = np.exp(-dt / tau_m)
    bm = m_inf * (1 - am)
    if use_h:
        ah = np.exp(-dt / tau_h)
        bh = h_inf * (1 - ah)
    if use_n:
        n_inf, tau_n = gating_n(V, params)
        an = np.exp(-dt / tau_n)
        bn = n_inf * (1 - an)
    # Initial gate states: explicit values win; otherwise steady state at V[0].
    m = m_inf[0] if m0 is None else float(m0)
    if use_h:
        h = h_inf[0] if h0 is None else float(h0)
    else:
        h = 1.0
    if use_n:
        n = n_inf[0] if n0 is None else float(n0)
    else:
        n = 1.0
    out = np.empty(len(V))
    for k in range(len(V)):
        m = am[k] * m + bm[k]
        if use_h:
            h = ah[k] * h + bh[k]
        if use_n:
            n = an[k] * n + bn[k]
        iv = g_bar * (m ** p) * (h ** ph) * (V[k] - E)
        if use_n:
            iv *= (n ** pn)
        if use_leak:
            iv += g_L * (V[k] - E_L)
        out[k] = iv
    return out


def honest_predict(V_mV, t_ms, params, mask, **sim_kw):
    """Honest segment predictions: simulate the FULL trace from t=0, then slice.

    V_mV, t_ms: full-trace arrays. mask: boolean array selecting the segment
    to score (e.g. the held-out validation window). sim_kw: passed through
    to simulate() (p, ph, pn, use_h, use_leak, use_n, m0/h0/n0, ...).
    Returns predictions on the masked segment only, with true gate history.
    This is the scoring Thursday's live test uses -- never score a segment
    by simulating it standalone.
    """
    mask = np.asarray(mask, dtype=bool)
    return simulate(V_mV, t_ms, params, **sim_kw)[mask]


def q0_base():
    import numpy as np
    return np.array([
        np.log(0.003), -8.6, np.log(1e-4), -50.0,
        -45.0, np.log(8.0), -25.0, np.log(7.0),
        np.log(0.1), np.log(4.0), -90.0, np.log(40.0),
        np.log(1.0), np.log(11.0), 20.0, np.log(40.0),
        # n gate: slow second activation (right-shifted, slower than m)
        -30.0, np.log(10.0),
        np.log(2.0), np.log(20.0), -50.0, np.log(40.0),
    ])


def bounds():
    # Box bounds (optimizer q-space; log() entries are log-space).
    # Wide enough to cover voltage protocols from -180..+80 mV: tau bell
    # centers/amplitudes and the leak ceiling must not choke on wider
    # excursions than the original -100..+20 step protocol.
    import numpy as np
    LB = np.array([np.log(1e-5), -30, np.log(1e-9), -90, -90, np.log(1), -80, np.log(1),
                   np.log(0.02), np.log(1e-6), -180, np.log(1), np.log(0.01), np.log(1e-6), -180, np.log(1),
                   -90, np.log(1), np.log(0.05), np.log(1e-6), -180, np.log(1)])
    UB = np.array([np.log(0.05), 15, np.log(0.01), 30, 0, np.log(30), 40, np.log(30),
                   np.log(20), np.log(200), 80, np.log(100), np.log(50), np.log(200), 80, np.log(100),
                   20, np.log(30), np.log(50), np.log(200), 80, np.log(100)])
    return LB, UB

