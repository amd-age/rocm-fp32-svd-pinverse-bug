"""
Isolate the DIMENSION effect: fp32 SVD error vs matrix size at a FIXED, moderate
condition number (below the catastrophe onset), CPU vs ROCm. With cond held
constant, any growth with N is a pure size effect, not conditioning.

Result: ROCm's fp32 svdvals error grows ~N^1.2 with size even at fixed cond, while
CPU stays nearly flat -- i.e. ROCm has a much larger error-growth-with-size
constant. (This is the benign baseline; it explains why ordinary matrices, whose
cond also grows with N, show a rising ROCm error. The catastrophic regime --
dense + ill-conditioned -- saturates and is then size-independent.)

Matrices: A = Q diag(s) Qᵀ with a geometric spectrum at the fixed cond.
Reference = fp64 SVD of the same fp32 matrix.

    python plot_baseline_size.py [cond]    # default cond=1e3 -> svd_error_vs_size_fixedcond.png
"""
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COND = float(sys.argv[1]) if len(sys.argv) > 1 else 1e3
SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
SEEDS = list(range(5))
DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print(f"torch={torch.__version__} hip={getattr(torch.version,'hip',None)} gpu={gpu}  cond={COND:.0e}\n")


def trial(N, seed):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(N, N, dtype=torch.float64, generator=g))
    s = (1.0 / COND) ** (torch.arange(N, dtype=torch.float64) / (N - 1))
    A = (Q * s) @ Q.T
    A32 = ((A + A.T) / 2).to(torch.float32)
    ref = torch.linalg.svdvals(A32.double())
    return {dev: ((torch.linalg.svdvals(A32.to(dev)).double().cpu() - ref).abs() / ref.abs()).mean().item()
            for dev in DEVS}


data = {dev: np.zeros((len(SIZES), len(SEEDS))) for dev in DEVS}
for i, N in enumerate(SIZES):
    for j, sd in enumerate(SEEDS):
        r = trial(N, sd)
        for dev in DEVS:
            data[dev][i, j] = r[dev]
    print(f"  N={N:>5}  " + "  ".join(f"{dev}={data[dev][i].mean():.2e}" for dev in DEVS))

# fit slope (log-log) for ROCm to report the growth exponent
x = np.array(SIZES, dtype=float)
if "cuda" in DEVS:
    p = np.polyfit(np.log(x), np.log(data["cuda"].mean(1)), 1)[0]
    print(f"\n  ROCm error ~ N^{p:.2f} at fixed cond={COND:.0e}")

colors = {"cpu": "tab:blue", "cuda": "tab:red"}
labels = {"cpu": "CPU fp32", "cuda": f"ROCm fp32 ({gpu})"}
fig, ax = plt.subplots(figsize=(8, 5.2))
for dev in DEVS:
    m = data[dev]
    ax.plot(x, m.mean(1), "-o", color=colors[dev], markersize=5, label=labels[dev])
    ax.fill_between(x, m.min(1), m.max(1), color=colors[dev], alpha=0.12)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(SIZES); ax.set_xticklabels(SIZES)
ax.set_xlabel("matrix size N  (N×N SPD matrix)")
ax.set_ylabel("mean singular-value relative error")
ax.set_title(f"fp32 torch.linalg.svdvals error vs size at FIXED condition number\n"
             f"geometric spectrum, cond={COND:.0e}  (isolates the dimension effect; "
             f"{len(SEEDS)} seeds)")
ax.legend(fontsize=9, loc="best")
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
out = "svd_error_vs_size_fixedcond.png"
fig.savefig(out, dpi=140)
print(f"saved {out}")
