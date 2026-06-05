"""
Plot SVD singular-value relative error vs matrix size, for SPD matrices with a
GEOMETRIC singular-value spectrum (fixed condition number), CPU vs ROCm.

Two figures:
  svd_error_vs_size.png       fp32  (the bug: ROCm fp32 is broken, CPU fp32 fine)
  svd_error_vs_size_fp64.png  fp64  (control: both backends accurate)

Matrix construction (SPD, so U = V = Q):
    s = (1/cond) ** (arange(N)/(N-1))        # geometric spectrum, 1 .. 1/cond
    Q, _ = qr(randn(N, N))                    # Haar-random orthogonal (fp64)
    A = Q diag(s) Qᵀ

References (ground truth singular values for what each backend was handed):
    fp32 backends get the fp32-rounded matrix -> ref = fp64 SVD of that fp32 matrix
         (isolates the SVD algorithm from fp32 input rounding).
    fp64 backends get the full fp64 matrix    -> ref = the prescribed spectrum s
         (exact by construction; fp64 SVD of A equals s to ~1e-15).

We plot the MEAN singular-value relative error over seeds (band = min/max).

    python plot_size.py            # cond=1e6, N=8..4096, 5 seeds
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
    """Return {(dtype, dev): mean_rel_err}."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(N, N, dtype=torch.float64, generator=g))
    s = (1.0 / cond) ** (torch.arange(N, dtype=torch.float64) / (N - 1))   # descending truth
    A64 = (Q * s) @ Q.T
    A64 = (A64 + A64.T) / 2
    A32 = A64.to(torch.float32)

    ref32 = torch.linalg.svdvals(A32.double())     # truth for the fp32 matrix
    ref64 = s                                       # truth for the fp64 matrix (exact)
    out = {}
    for dev in DEVS:
        S32 = torch.linalg.svdvals(A32.to(dev)).double().cpu()
        out[("fp32", dev)] = ((S32 - ref32).abs() / ref32.abs()).mean().item()
        S64 = torch.linalg.svdvals(A64.to(dev)).double().cpu()
        out[("fp64", dev)] = ((S64 - ref64).abs() / ref64.abs()).mean().item()
    return out


keys = [(dt, dev) for dt in ("fp32", "fp64") for dev in DEVS]
acc = {k: np.zeros((len(SIZES), len(SEEDS))) for k in keys}
for i, N in enumerate(SIZES):
    for j, sd in enumerate(SEEDS):
        r = trial(N, COND, sd)
        for k in keys:
            acc[k][i, j] = r[k]
    msg = "  ".join(f"{dt}/{dev}={acc[(dt,dev)][i].mean():.2e}" for dt, dev in keys)
    print(f"  N={N:>5}  {msg}")

x = np.array(SIZES)
colors = {"cpu": "tab:blue", "cuda": "tab:red"}
labels = {"cpu": "CPU", "cuda": f"ROCm ({gpu})"}


def make_plot(dtype, fname, title_extra):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for dev in DEVS:
        m = acc[(dtype, dev)]
        ax.plot(x, m.mean(1), "-o", color=colors[dev], markersize=5,
                label=f"{labels[dev]} {dtype} (mean σ err)")
        ax.fill_between(x, m.min(1), m.max(1), color=colors[dev], alpha=0.12)
    ax.axhline(0.1, color="gray", ls=":", lw=1)
    ax.text(x[0], 0.11, "10% error", color="gray", fontsize=8, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(x)
    ax.set_xlabel("matrix size N  (N×N SPD matrix)")
    ax.set_ylabel("mean singular-value relative error")
    ax.set_title(f"{dtype} torch.linalg.svdvals error vs size  ({title_extra})\n"
                 f"geometric spectrum, cond={COND:.0e}  (band = min/max over {len(SEEDS)} seeds)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    print(f"saved {fname}")


make_plot("fp32", "svd_error_vs_size.png", "the bug")
make_plot("fp64", "svd_error_vs_size_fp64.png", "control: fp64 is fine on both")
