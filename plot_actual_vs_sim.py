"""Plot actual vs simulated current from selection_results.json.

For each structure (top-3 by valid RMSE + the m^1 h^1 baseline): left = trace
overlay on the held-out segment, right = parity scatter (actual vs simulated)
with the y=x line. Residuals live in the parity plot's vertical spread.

Run:  python plot_actual_vs_sim.py [selection_results.json] [data.txt] [out.png]
Needs: numpy, matplotlib, hh_core.py
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hh_core import simulate, unpack

res_path = sys.argv[1] if len(sys.argv) > 1 else 'selection_results.json'
data_path = sys.argv[2] if len(sys.argv) > 2 else 'CookLab1UnknownChannelExtra.txt'
out_path = sys.argv[3] if len(sys.argv) > 3 else 'actual_vs_sim.png'

res = json.load(open(res_path))
d = np.loadtxt(data_path)
t, V, I = d[:, 0], d[:, 1], d[:, 2]
n = len(t)
nv = int(n * 0.15)
va = np.zeros(n, bool)
va[-nv:] = True

agg = sorted(res['aggregate'], key=lambda r: r['va_min'])
picks = agg[:20]
if not any(a['name'] == 'm^1 h^1' for a in picks):
    picks.append(next(a for a in agg if a['name'] == 'm^1 h^1'))

fig, axes = plt.subplots(len(picks), 2, figsize=(12, 3.2 * len(picks)),
                         sharex=False)
if len(picks) == 1:
    axes = axes.reshape(1, -1)
for ax_row, a in zip(axes, picks):
    q = unpack(np.array(a['q_min']))
    pred = simulate(V[va], t[va], q, p=a['pm'], ph=a['ph'], pn=a['pn'],
                    use_h=a['use_h'], use_leak=a['use_leak'], use_n=a['use_n'])
    y = I[va]
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))

    ax = ax_row[0]
    ax.plot(t[va], y, 'k-', lw=0.8, alpha=0.7, label='actual')
    ax.plot(t[va], pred, 'r-', lw=0.8, alpha=0.7, label='simulated')
    ax.set_title(f"{a['name']}: valid RMSE={rmse:.5f}  R²={a['r2_mean']:.4f}")
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('I')
    ax.legend(fontsize=8)

    ax = ax_row[1]
    lo = min(y.min(), pred.min())
    hi = max(y.max(), pred.max())
    ax.scatter(y, pred, s=2, alpha=0.3, c='steelblue')
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1, label='y = x')
    ax.set_title(f"{a['name']}: parity (actual vs simulated)")
    ax.set_xlabel('actual I')
    ax.set_ylabel('simulated I')
    ax.legend(fontsize=8)
    ax.set_aspect('equal', adjustable='box')

fig.suptitle(f"actual vs simulated on held-out segment "
             f"(fits from {res['data'].split('/')[-1]})", fontsize=12)
fig.tight_layout()
fig.savefig(out_path, dpi=150)
print(f"saved to {out_path}")
