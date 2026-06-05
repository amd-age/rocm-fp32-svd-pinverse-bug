"""
Plot fp32 SVD singular-value relative error vs CONDITION NUMBER, CPU vs ROCm,
for SPD matrices with a GEOMETRIC singular-value spectrum.

For a fixed (geometric) spectrum the error rises with condition number on both
backends -- but ROCm crosses 10% near cond ~1e4, while CPU stays accurate until
cond ~1e8, i.e. ~3-4 orders of magnitude further. (The rise itself is expected:
the relative error of the small singular values is floored at ~eps*cond by Weyl's
theorem; the point is that ROCm sits far above that fundamental floor.)

Matrices are SPD: A = Q diag(s) Qᵀ, Q Haar-random orthogonal (fp64), cast to fp32.
Reference singular values = fp64 SVD of the same fp32 matrix.

    python plot_cond.py            # N=512, cond 1e1..1e9, 3 seeds -> svd_error_vs_cond.png
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 512
CONDS = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
SEEDS = list(range(3))
DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print(f"torch={torch.__version__} hip={getattr(torch.version,'hip',None)} gpu={gpu}")


def err(cond, seed, dev):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(N, N, dtype=torch.float64, generator=g))
    s = (1.0 / cond) ** (torch.arange(N, dtype=torch.float64) / (N - 1))   # geometric
    A = (Q * s) @ Q.T
    A32 = ((A + A.T) / 2).to(torch.float32)
    ref = torch.linalg.svdvals(A32.double())
    S = torch.linalg.svdvals(A32.to(dev)).double().cpu()
    return ((S - ref).abs() / ref.abs()).mean().item()


def curve(dev):
    out = np.zeros((len(CONDS), len(SEEDS)))
    for i, c in enumerate(CONDS):
        for j, sd in enumerate(SEEDS):
            out[i, j] = err(c, sd, dev)
    return out


data = {dev: curve(dev) for dev in DEVS}
for dev in DEVS:
    print(f"  {dev:>4}: " + " ".join(f"{data[dev][i].mean():.1e}" for i in range(len(CONDS))))

x = np.array(CONDS)
colors = {"cpu": "tab:blue", "cuda": "tab:red"}
labels = {"cpu": "CPU fp32", "cuda": f"ROCm fp32 ({gpu})"}

fig, ax = plt.subplots(figsize=(8, 5.2))
for dev in DEVS:
    m = data[dev]
    ax.plot(x, m.mean(1), "-o", color=colors[dev], markersize=6, label=labels[dev])
    ax.fill_between(x, m.min(1), m.max(1), color=colors[dev], alpha=0.12)
ax.axhline(0.1, color="gray", ls=":", lw=1)
ax.text(x[0], 0.12, "10% error", color="gray", fontsize=8, va="bottom")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("condition number  cond(A)")
ax.set_ylabel("mean singular-value relative error")
ax.set_title(f"fp32 torch.linalg.svdvals error vs condition number\n"
             f"geometric spectrum, N={N} SPD  (band = min/max over {len(SEEDS)} seeds)")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig("svd_error_vs_cond.png", dpi=140)
print("saved svd_error_vs_cond.png")
