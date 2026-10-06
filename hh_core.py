"""Core HH model: shared by fitting scripts and the deliverable hh_model.py.

Model: I = g_bar * m^p * h * (V - E) + g_L * (V - E_L)
  m_inf(V) = 1/(1+exp(-(V-Vm)/km))        (activation)
  h_inf(V) = 1/(1+exp(+(V-Vh)/kh))        (inactivation)
  tau_m(V) = tm_min + tm_amp * exp(-((V-tm_c)/tm_w)^2)
  tau_h(V) = th_min + th_amp * exp(-((V-th_c)/th_w)^2)
State update is the exact solution over each sample interval (piecewise-constant V):
  X <- X + (Xinf(V) - X) * (1 - exp(-dt/tau(V)))
"""
import numpy as np

PNAMES = ['g_bar', 'E', 'g_L', 'E_L', 'Vm', 'km', 'Vh', 'kh',
          'tm_min', 'tm_amp', 'tm_c', 'tm_w', 'th_min', 'th_amp', 'th_c', 'th_w']
DT = 0.25  # ms, sampling of the data file


def unpack(q):
    q = np.asarray(q, dtype=float)
    g_bar = np.exp(q[0]); E = q[1]; g_L = np.exp(q[2]); E_L = q[3]
    Vm = q[4]; km = np.exp(q[5]); Vh = q[6]; kh = np.exp(q[7])
    tm_min = np.exp(q[8]); tm_amp = np.exp(q[9]); tm_c = q[10]; tm_w = np.exp(q[11])
    th_min = np.exp(q[12]); th_amp = np.exp(q[13]); th_c = q[14]; th_w = np.exp(q[15])
    return (g_bar, E, g_L, E_L, Vm, km, Vh, kh,
            tm_min, tm_amp, tm_c, tm_w, th_min, th_amp, th_c, th_w)


def pack(params):
    (g_bar, E, g_L, E_L, Vm, km, Vh, kh,
     tm_min, tm_amp, tm_c, tm_w, th_min, th_amp, th_c, th_w) = params
    return np.array([np.log(g_bar), E, np.log(g_L), E_L, Vm, np.log(km), Vh, np.log(kh),
                     np.log(tm_min), np.log(tm_amp), tm_c, np.log(tm_w),
                     np.log(th_min), np.log(th_amp), th_c, np.log(th_w)])


def gating(V, params):
    """Return (m_inf, tau_m, h_inf, tau_h) arrays for voltage array V (mV)."""
    (g_bar, E, g_L, E_L, Vm, km, Vh, kh,
     tm_min, tm_amp, tm_c, tm_w, th_min, th_amp, th_c, th_w) = params
    V = np.asarray(V, dtype=float)
    m_inf = 1.0 / (1.0 + np.exp(-(V - Vm) / km))
    tau_m = tm_min + tm_amp * np.exp(-((V - tm_c) / tm_w) ** 2)
    h_inf = 1.0 / (1.0 + np.exp((V - Vh) / kh))
    tau_h = th_min + th_amp * np.exp(-((V - th_c) / th_w) ** 2)
    return m_inf, tau_m, h_inf, tau_h


def simulate(V_mV, t_ms, params, p=1, use_h=True, use_leak=True, dt=None):
    """Predict current for a voltage-clamp waveform.

    V_mV, t_ms: array-likes of equal length (t in ms, uniformly sampled).
    params: dict or sequence of the 16 physical parameters (see PNAMES).
    Returns: predicted current array (same units as fitted data).
    Initial gate states = steady state at V_mV[0].
    """
    if isinstance(params, dict):
        params = [params[k] for k in PNAMES]
    V = np.asarray(V_mV, dtype=float)
    t = np.asarray(t_ms, dtype=float)
    if dt is None:
        dt = float(np.median(np.diff(t)))
    g_bar, E, g_L, E_L, Vm, km, Vh, kh, tm_min, tm_amp, tm_c, tm_w, th_min, th_amp, th_c, th_w = params
    m_inf, tau_m, h_inf, tau_h = gating(V, params)
    am = np.exp(-dt / tau_m)
    bm = m_inf * (1 - am)
    if use_h:
        ah = np.exp(-dt / tau_h)
        bh = h_inf * (1 - ah)
    m = m_inf[0]
    h = h_inf[0] if use_h else 1.0
    out = np.empty(len(V))
    for k in range(len(V)):
        m = am[k] * m + bm[k]
        if use_h:
            h = ah[k] * h + bh[k]
        iv = g_bar * (m ** p) * h * (V[k] - E)
        if use_leak:
            iv += g_L * (V[k] - E_L)
        out[k] = iv
    return out


def q0_base():
    import numpy as np
    return np.array([
        np.log(0.003), -8.6, np.log(1e-4), -50.0,
        -45.0, np.log(8.0), -25.0, np.log(7.0),
        np.log(0.1), np.log(4.0), -90.0, np.log(40.0),
        np.log(1.0), np.log(11.0), 20.0, np.log(40.0),
    ])


def bounds():
    import numpy as np
    LB = np.array([np.log(1e-5), -30, np.log(1e-9), -90, -90, np.log(1), -80, np.log(1),
                   np.log(0.02), np.log(1e-6), -120, np.log(1), np.log(0.05), np.log(1e-6), -120, np.log(1)])
    UB = np.array([np.log(0.05), 15, np.log(0.002), 30, 0, np.log(30), 40, np.log(30),
                   np.log(20), np.log(50), 60, np.log(100), np.log(50), np.log(100), 60, np.log(100)])
    return LB, UB
