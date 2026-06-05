"""
Control plot: fp32 SVD error vs matrix size for ORDINARY (well-conditioned) random
matrices -- iid Gaussian entries, no spectrum engineering. This is the common case;
ROCm fp32 should be fine here (the bug needs ill-conditioned + dense spectra).

    A = randn(N, N)              (cond grows only ~linearly with N)
    reference singular values = fp64 SVD of the same fp32 matrix

Both CPU and ROCm fp32 should sit far below the 10% line at every size.

    python plot_normal_size.py        # N=8..4096, 5 seeds -> svd_error_vs_size_normal.png
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
SEEDS = list(range(5))
DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print(f"torch={torch.__version__} hip={getattr(torch.version,'hip',None)} gpu={gpu}\n")


def trial(N, seed):
    g = torch.Generator().manual_seed(seed)
    A32 = torch.randn(N, N, dtype=torch.float64, generator=g).to(torch.float32)  # ordinary random
    ref = torch.linalg.svdvals(A32.double())
    cond = (ref[0] / ref[-1]).item()
    out = {"cond": cond}
    for dev in DEVS:
        S = torch.linalg.svdvals(A32.to(dev)).double().cpu()
        out[dev] = ((S - ref).abs() / ref.abs()).mean().item()
    return out


data = {dev: np.zeros((len(SIZES), len(SEEDS))) for dev in DEVS}
conds = np.zeros((len(SIZES), len(SEEDS)))
for i, N in enumerate(SIZES):
    for j, sd in enumerate(SEEDS):
        r = trial(N, sd)
        conds[i, j] = r["cond"]
        for dev in DEVS:
            data[dev][i, j] = r[dev]
    line = "  ".join(f"{dev}={data[dev][i].mean():.2e}" for dev in DEVS)
    print(f"  N={N:>5}  cond~{conds[i].mean():.1e}  {line}")

x = np.array(SIZES)
colors = {"cpu": "tab:blue", "cuda": "tab:red"}
labels = {"cpu": "CPU fp32", "cuda": f"ROCm fp32 ({gpu})"}

fig, ax = plt.subplots(figsize=(8, 5.2))
for dev in DEVS:
    m = data[dev]
    ax.plot(x, m.mean(1), "-o", color=colors[dev], markersize=5, label=labels[dev])
    ax.fill_between(x, m.min(1), m.max(1), color=colors[dev], alpha=0.12)
ax.axhline(0.1, color="gray", ls=":", lw=1)
ax.text(x[0], 0.12, "10% error", color="gray", fontsize=8, va="bottom")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(x)
ax.set_ylim(top=1.0)
ax.set_xlabel("matrix size N  (N×N matrix)")
ax.set_ylabel("mean singular-value relative error")
ax.set_title("fp32 torch.linalg.svdvals error vs size, ORDINARY random matrices\n"
             f"iid Gaussian (well-conditioned, cond ~ N)  (band = min/max over {len(SEEDS)} seeds)")
ax.legend(fontsize=9, loc="best")
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
out = "svd_error_vs_size_normal.png"
fig.savefig(out, dpi=140)
print(f"\nsaved {out}")
