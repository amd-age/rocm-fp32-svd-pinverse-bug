"""
Plot fp32 SVD singular-value relative error vs matrix size, for SPD matrices with
a GEOMETRIC singular-value spectrum (fixed condition number). CPU fp32 vs ROCm
fp32, each compared to the fp64 SVD of the identical fp32 matrix.

Matrix construction (SPD, so U = V = Q):
    s = (1/cond) ** (arange(N)/(N-1))        # geometric spectrum, 1 .. 1/cond
    Q, _ = qr(randn(N, N))                    # Haar-random orthogonal (fp64)
    A = Q diag(s) Qᵀ  -> cast to fp32

Several seeds per N; we plot the mean over seeds with a min/max band.

    python plot_size.py            # cond=1e6, N=8..4096, 5 seeds -> svd_error_vs_size.png
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COND = 1e6
SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
SEEDS = list(range(5))
DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print(f"torch={torch.__version__} hip={getattr(torch.version,'hip',None)} gpu={gpu}")


def trial(N, cond, seed):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(N, N, dtype=torch.float64, generator=g))
    s = (1.0 / cond) ** (torch.arange(N, dtype=torch.float64) / (N - 1))
    A = ((Q * s) @ Q.T)
    A = ((A + A.T) / 2).to(torch.float32)
    ref = torch.linalg.svdvals(A.double())
    out = {}
    for dev in DEVS:
        S = torch.linalg.svdvals(A.to(dev)).double().cpu()
        rel = (S - ref).abs() / ref.abs()
        out[dev] = (rel.mean().item(), rel.max().item())
    return out


# collect: data[dev]['mean'/'max'] -> array [len(SIZES), len(SEEDS)]
data = {d: {"mean": np.zeros((len(SIZES), len(SEEDS))),
            "max": np.zeros((len(SIZES), len(SEEDS)))} for d in DEVS}
for i, N in enumerate(SIZES):
    for j, sd in enumerate(SEEDS):
        r = trial(N, COND, sd)
        for d in DEVS:
            data[d]["mean"][i, j], data[d]["max"][i, j] = r[d]
    line = "  ".join(f"{d}: mean={data[d]['mean'][i].mean():.2e}" for d in DEVS)
    print(f"  N={N:>5}  {line}")

x = np.array(SIZES)
colors = {"cpu": "tab:blue", "cuda": "tab:red"}
labels = {"cpu": "CPU fp32", "cuda": f"ROCm fp32 ({gpu})"}

fig, ax = plt.subplots(figsize=(8, 5.2))
for d in DEVS:
    for stat, ls, alpha in [("mean", "-", 1.0), ("max", "--", 0.6)]:
        m = data[d][stat]
        ax.plot(x, m.mean(1), ls, color=colors[d], alpha=alpha,
                marker="o" if stat == "mean" else "^", markersize=5,
                label=f"{labels[d]} ({'mean' if stat=='mean' else 'max'} σ err)")
        ax.fill_between(x, m.min(1), m.max(1), color=colors[d], alpha=0.12)

ax.axhline(0.1, color="gray", ls=":", lw=1)
ax.text(x[0], 0.11, "10% error", color="gray", fontsize=8, va="bottom")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.set_xlabel("matrix size N  (N×N SPD matrix)")
ax.set_ylabel("singular-value relative error  vs fp64")
ax.set_title(f"fp32 torch.linalg.svdvals error vs size\n"
             f"geometric spectrum, cond={COND:.0e}  (band = min/max over {len(SEEDS)} seeds)")
ax.legend(fontsize=8, loc="center left")
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
out = "svd_error_vs_size.png"
fig.savefig(out, dpi=140)
print(f"saved {out}")
