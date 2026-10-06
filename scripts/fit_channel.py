"""Fit a one-gate ion channel model to voltage-clamp data.

Usage: python fit_channel.py data1.txt [data2.txt ...]
Each file: tab-separated, no header, columns time (ms), Vm (mV), I.
Each file is assumed to hold at one voltage, then step once to another.
"""
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

N_STEPS = 5000

# --- load data and find the step in each file ---
sweeps = []
for fname in sys.argv[1:]:
    t, v, i = np.loadtxt(fname, delimiter="\t", unpack=True) # loading time, voltage, current
    k = np.flatnonzero(np.diff(v) != 0)[0] + 1   # first sample at the new voltage 
    sweeps.append(dict(name=fname, t=t, v=v, i=i, t_step=t[k], v_hold=v[0], v_step=v[k]))

all_v = sorted({s["v_hold"] for s in sweeps} | {s["v_step"] for s in sweeps})
step_v = sorted({s["v_step"] for s in sweeps})
x_idx = {v: n for n, v in enumerate(all_v)}
tau_idx = {v: n for n, v in enumerate(step_v)}

# Scale current so the loss is O(1)
i_scale = np.concatenate([s["i"] for s in sweeps]).std()

# --- parameters (unconstrained) ---
er0 = 0.0
v_all = np.concatenate([s["v"] for s in sweeps])
g0 = np.mean(np.abs(np.concatenate([s["i"] for s in sweeps]))) / np.mean(np.abs(v_all - er0)) / 0.5
log_gbar = torch.tensor(np.log(g0), requires_grad=True)
Er = torch.tensor(er0, requires_grad=True)
xinf_raw = torch.zeros(len(all_v), requires_grad=True)                       # sigmoid(0) = 0.5
log_tau = torch.full((len(step_v),), float(np.log(5.0)), requires_grad=True)  # 5 ms


def model(s):
    t = torch.tensor(s["t"])
    v = torch.tensor(s["v"])
    xinf = torch.sigmoid(xinf_raw)
    x0 = xinf[x_idx[s["v_hold"]]]
    x1 = xinf[x_idx[s["v_step"]]]
    tau = torch.exp(log_tau[tau_idx[s["v_step"]]])
    ts = torch.clamp(t - s["t_step"], min=0.0)          # 0 before the step -> X = X0
    x = x0 + (x1 - x0) * (1 - torch.exp(-ts / tau))
    return torch.exp(log_gbar) * x * (v - Er)


# Er is in mV, so give it a larger learning rate than the log/logit params
opt = torch.optim.Adam([
    {"params": [log_gbar, xinf_raw, log_tau], "lr": 0.02},
    {"params": [Er], "lr": 0.5},
])
targets = [torch.tensor(s["i"]) for s in sweeps]
for step in range(N_STEPS):
    opt.zero_grad()
    loss = sum(torch.mean(((model(s) - y) / i_scale) ** 2) for s, y in zip(sweeps, targets)) / len(sweeps)
    loss.backward()
    opt.step()
    if step % 500 == 0:
        print(f"step {step:5d}  loss {loss.item():.4e}")

# --- results ---
print(f"\nfinal loss (scaled MSE): {loss.item():.4e}")
print(f"gbar = {torch.exp(log_gbar).item():.6g}")
print(f"Er   = {Er.item():.3f} mV")
for v, x in zip(all_v, torch.sigmoid(xinf_raw).tolist()):
    print(f"Xinf({v:g} mV) = {x:.4f}")
for v, tau in zip(step_v, torch.exp(log_tau).tolist()):
    print(f"tau({v:g} mV)  = {tau:.4f} ms")

with torch.no_grad():
    for s in sweeps:
        plt.plot(s["t"], s["i"], ".", ms=2, label=f"measured ({s['name']})")
        plt.plot(s["t"], model(s).numpy(), "-", label=f"model (step to {s['v_step']:g} mV)")
plt.xlabel("time (ms)")
plt.ylabel("current")
plt.legend()
plt.tight_layout()
plt.show()
