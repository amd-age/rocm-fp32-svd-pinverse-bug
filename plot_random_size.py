"""
Like plot_size.py, but the singular values are RANDOMIZED (drawn at random) rather
than placed on a deterministic geometric law -- to show the bug is not an artifact
of the exact spectrum, only of having many singular values densely spread over a
wide range. Plots fp32 SVD error vs matrix size, CPU vs ROCm.

Spectrum: log-uniform random in [1/cond, 1] (so the condition number is fixed at
`cond` and the values are dense across the range, but randomly spaced). Pass
`uniform` for linear-uniform instead.

    A = Q diag(s) Qᵀ,  Q Haar-random orthogonal (fp64),  cast to fp32
    reference singular values = fp64 SVD of the same fp32 matrix

VERDICT NOTE
------------
We do NOT call a result "broken" by an absolute error threshold. At high condition
number fp32 cannot represent the small singular values to full precision on ANY
backend -- that is a property of fp32, not of the hardware (by Weyl's theorem the
relative error of the small singular values is floored at ~eps_fp32 * cond, and CPU
fp32 hits that floor too). The meaningful signal is whether ROCm does much WORSE
than the best achievable fp32 result, for which CPU fp32 is the reference. So the
verdict is based on the excess ratio  rocm_error / cpu_error.

    python plot_random_size.py [cond] [dist]   # defaults: cond=1e6, dist=loguniform
"""
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COND = float(sys.argv[1]) if len(sys.argv) > 1 else 1e6
DIST = sys.argv[2] if len(sys.argv) > 2 else "loguniform"
SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
SEEDS = list(range(5))
DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print(f"torch={torch.__version__} hip={getattr(torch.version,'hip',None)} gpu={gpu}")
print(f"random spectrum: dist={DIST}  cond={COND:.0e}\n")


def random_spectrum(N, cond, dist, gen):
    """Random singular values with exact condition number `cond` (max=1, min=1/cond)."""
    u = torch.rand(N, dtype=torch.float64, generator=gen)
    if dist == "loguniform":
        s = cond ** (-u)                       # log-uniform in [1/cond, 1]
    elif dist == "uniform":
        s = (1.0 / cond) + (1.0 - 1.0 / cond) * u   # linear-uniform in [1/cond, 1]
    else:
        raise ValueError(dist)
    s = s.sort(descending=True).values
    s[0], s[-1] = 1.0, 1.0 / cond              # anchor endpoints -> exact cond
    return s


def trial(N, cond, seed):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(N, N, dtype=torch.float64, generator=g))
    s = random_spectrum(N, cond, DIST, g)
    A = (Q * s) @ Q.T
    A32 = ((A + A.T) / 2).to(torch.float32)
    ref = torch.linalg.svdvals(A32.double())
    out = {}
    for dev in DEVS:
        S = torch.linalg.svdvals(A32.to(dev)).double().cpu()
        out[dev] = ((S - ref).abs() / ref.abs()).mean().item()
    return out


def verdict(cpu_mean, rocm_mean):
    """Relative to the achievable fp32 baseline (CPU), not an absolute threshold.
    CPU fp32 = what fp32 can achieve on this matrix; the bug is ROCm doing much
    worse WHILE CPU is still accurate (so fp32 itself was not the limit)."""
    ratio = rocm_mean / cpu_mean if cpu_mean > 0 else float("inf")
    if cpu_mean > 0.1:
        return f"fp32 floor on BOTH (cond too high; not hardware). ROCm {ratio:>4.0f}x worse"
    if rocm_mean > 1e-3 and ratio > 5:
        return f"ROCm {ratio:>5.0f}x worse than CPU fp32  <-- backend error"
    if ratio > 5:
        return f"ROCm {ratio:>5.0f}x worse (but both small; benign)"
    return "ROCm ~ CPU (matrix easy for fp32)"


data = {dev: np.zeros((len(SIZES), len(SEEDS))) for dev in DEVS}
for i, N in enumerate(SIZES):
    for j, sd in enumerate(SEEDS):
        r = trial(N, COND, sd)
        for dev in DEVS:
            data[dev][i, j] = r[dev]
    cpu_m = data["cpu"][i].mean()
    line = f"  N={N:>5}  cpu={cpu_m:.2e}"
    if "cuda" in DEVS:
        rocm_m = data["cuda"][i].mean()
        line += f"  rocm={rocm_m:.2e}  | {verdict(cpu_m, rocm_m)}"
    print(line)

x = np.array(SIZES)
colors = {"cpu": "tab:blue", "cuda": "tab:red"}
labels = {"cpu": "CPU fp32", "cuda": f"ROCm fp32 ({gpu})"}

fig, ax = plt.subplots(figsize=(8, 5.2))
for dev in DEVS:
    m = data[dev]
    ax.plot(x, m.mean(1), "-o", color=colors[dev], markersize=5, label=labels[dev])
    ax.fill_between(x, m.min(1), m.max(1), color=colors[dev], alpha=0.12)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(x)
ax.set_xlabel("matrix size N  (N×N SPD matrix)")
ax.set_ylabel("mean singular-value relative error")
ax.set_title(f"fp32 torch.linalg.svdvals error vs size, RANDOM singular values\n"
             f"{DIST} spectrum, cond={COND:.0e}  (band = min/max over {len(SEEDS)} seeds)")
ax.legend(fontsize=9, loc="best")
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
out = "svd_error_vs_size_random.png"
fig.savefig(out, dpi=140)
print(f"\nsaved {out}")
