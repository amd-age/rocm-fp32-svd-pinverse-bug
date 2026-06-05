"""
Plot fp32 SVD singular-value relative error vs CONDITION NUMBER, for several
spectral SHAPES, on ROCm. The question: does condition number predict the error?

Answer (the plot): NO, not by itself. At the SAME condition number, the error
depends on the spectral shape:
  - geometric / flat_tail (many singular values densely spread over the range):
    error climbs with cond and crosses 10% around cond ~1e4.
  - outlier (a well-conditioned bulk plus one isolated tiny singular value): the
    cond is high but the error stays low -- ROCm handles it fine.
CPU fp32 on the geometric spectrum is shown (dashed) as a reference: it stays
accurate ~3-4 orders of magnitude further in cond than ROCm.

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
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
HAS_GPU = torch.cuda.is_available()
print(f"torch={torch.__version__} hip={getattr(torch.version,'hip',None)} gpu={gpu}")


def spectrum(kind, N, cond):
    i = torch.arange(N, dtype=torch.float64)
    if kind == "geometric":                       # dense smooth decay across whole range
        return (1.0 / cond) ** (i / (N - 1))
    if kind == "flat_tail":                        # signal decay + clustered floor (mimics real)
        nsig = max(2, int(N * 0.4))
        s = torch.full((N,), 1.0 / cond, dtype=torch.float64)
        s[:nsig] = (1.0 / cond) ** (torch.arange(nsig, dtype=torch.float64) / (nsig - 1))
        return s
    if kind == "outlier":                          # well-conditioned bulk + ONE tiny outlier
        s = torch.full((N,), 1.0, dtype=torch.float64)
        s[: N - 1] = (0.1) ** (torch.arange(N - 1, dtype=torch.float64) / (N - 2))  # bulk cond 10
        s[-1] = 1.0 / cond                          # cond driven by this single value
        return s
    raise ValueError(kind)


def err(kind, cond, seed, dev):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(N, N, dtype=torch.float64, generator=g))
    s = spectrum(kind, N, cond)
    A = (Q * s) @ Q.T
    A32 = ((A + A.T) / 2).to(torch.float32)
    ref = torch.linalg.svdvals(A32.double())
    S = torch.linalg.svdvals(A32.to(dev)).double().cpu()
    return ((S - ref).abs() / ref.abs()).mean().item()


def curve(kind, dev):
    out = np.zeros((len(CONDS), len(SEEDS)))
    for i, c in enumerate(CONDS):
        for j, sd in enumerate(SEEDS):
            out[i, j] = err(kind, c, sd, dev)
    return out


SHAPES = ["geometric", "flat_tail", "outlier"]
rocm = {k: curve(k, "cuda") for k in SHAPES} if HAS_GPU else {}
cpu_geo = curve("geometric", "cpu")
for k in SHAPES:
    if HAS_GPU:
        print(f"  rocm/{k:>10}: " + " ".join(f"{rocm[k][i].mean():.1e}" for i in range(len(CONDS))))
print(f"  cpu /geometric : " + " ".join(f"{cpu_geo[i].mean():.1e}" for i in range(len(CONDS))))

x = np.array(CONDS)
style = {"geometric": ("tab:red", "o", "ROCm fp32  geometric (dense decay)"),
         "flat_tail": ("tab:orange", "s", "ROCm fp32  flat_tail (decay + floor)"),
         "outlier":   ("tab:green", "^", "ROCm fp32  outlier (1 isolated small σ)")}

fig, ax = plt.subplots(figsize=(8.2, 5.4))
if HAS_GPU:
    for k in SHAPES:
        c, m, lab = style[k]
        ax.plot(x, rocm[k].mean(1), "-", color=c, marker=m, markersize=6, label=lab)
        ax.fill_between(x, rocm[k].min(1), rocm[k].max(1), color=c, alpha=0.12)
ax.plot(x, cpu_geo.mean(1), "--", color="tab:blue", marker="o", markersize=5,
        label="CPU fp32  geometric (reference)")

ax.axhline(0.1, color="gray", ls=":", lw=1)
ax.text(x[0], 0.12, "10% error", color="gray", fontsize=8, va="bottom")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("condition number  cond(A)")
ax.set_ylabel("mean singular-value relative error")
ax.set_title(f"fp32 torch.linalg.svdvals error vs condition number, by spectral shape\n"
             f"N={N} SPD  (band = min/max over {len(SEEDS)} seeds; gpu={gpu})")
ax.legend(fontsize=8.5, loc="lower right")
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig("svd_error_vs_cond.png", dpi=140)
print("saved svd_error_vs_cond.png")
