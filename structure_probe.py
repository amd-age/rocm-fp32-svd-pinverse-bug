"""
What matrix STRUCTURE triggers the ROCm fp32 SVD error?

We synthesize SPD matrices A = Q diag(s) Qᵀ (Q Haar-random orthogonal, built in
fp64) with a *prescribed* spectrum s, then compare fp32 singular values from CPU
vs the ROCm GPU, each against the fp64 SVD of the identical fp32 matrix (so the
only difference is the fp32 SVD algorithm/backend).

Hypothesis: the trigger is a CLUSTER of nearly-equal small singular values (as in
the real matrix: one huge value + a flat tail at the regularization floor), not
the condition number alone.

Sweeps (pick with argv, default 'shape'):
    python structure_probe.py shape   # spectrum shape at fixed N, cond
    python structure_probe.py cond    # condition-number sweep, triggering shape
    python structure_probe.py size    # matrix-size sweep, triggering shape
    python structure_probe.py all

Metric per backend: mean / max relative error of singular values vs fp64, and the
fraction of singular values off by >10%. The headline is excess = cuda_mean/cpu_mean.
"""
import sys
import torch

torch.manual_seed(0)
DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
print(f"torch={torch.__version__} hip={getattr(torch.version,'hip',None)} "
      f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}\n")


def make_spd(eigs):
    """A = Q diag(eigs) Qᵀ with Q Haar-random orthogonal, all in fp64."""
    N = len(eigs)
    Q, _ = torch.linalg.qr(torch.randn(N, N, dtype=torch.float64))
    A = (Q * eigs) @ Q.T
    return (A + A.T) / 2


def spectrum(kind, N, kappa, flat_frac=0.5):
    """Descending positive spectrum, s_max=1, s_min=1/kappa."""
    smin = 1.0 / kappa
    i = torch.arange(N, dtype=torch.float64)
    if kind == "geometric":                       # smooth full-range decay
        return (smin) ** (i / (N - 1))
    if kind == "linear":                          # uniform spacing
        return torch.linspace(1.0, smin, N, dtype=torch.float64)
    if kind == "flat_tail":                       # signal decay + clustered floor (mimics real)
        nflat = int(N * flat_frac)
        nsig = N - nflat
        s = torch.full((N,), smin, dtype=torch.float64)
        s[:nsig] = (smin * 10) ** (torch.arange(nsig, dtype=torch.float64) / max(nsig - 1, 1))
        return s
    if kind == "two_cluster":                     # half at top, half at bottom (maximal clustering)
        s = torch.full((N,), smin, dtype=torch.float64)
        s[: N // 2] = 1.0
        return s
    if kind == "one_spike":                        # one huge value, rest a tight cluster at floor
        s = torch.full((N,), smin, dtype=torch.float64)
        s[0] = 1.0
        return s
    raise ValueError(kind)


def measure(A64):
    """Return {dev: (mean_relerr, max_relerr, frac>10%)} vs fp64 SVD of the fp32 matrix."""
    A32 = A64.to(torch.float32)
    ref = torch.linalg.svdvals(A32.double())      # truth for THIS fp32 matrix
    out = {}
    for dev in DEVS:
        S = torch.linalg.svdvals(A32.to(dev)).double().cpu()
        rel = (S - ref).abs() / ref.abs()
        out[dev] = (rel.mean().item(), rel.max().item(), (rel > 0.1).float().mean().item())
    return out


def row(label, m):
    cpu = m["cpu"]
    s = f"  {label:<26} | cpu: mean={cpu[0]:.2e} max={cpu[1]:.2e} f>10%={cpu[2]:.2f}"
    if "cuda" in m:
        g = m["cuda"]
        excess = g[0] / cpu[0] if cpu[0] > 0 else float("inf")
        s += f"  || rocm: mean={g[0]:.2e} max={g[1]:.2e} f>10%={g[2]:.2f}  EXCESS={excess:.0f}x"
    return s


def sweep_shape():
    N, kappa = 1025, 1e6
    print(f"[SHAPE]  N={N}  cond={kappa:.0e}  (which spectral shape triggers it?)")
    for kind in ["geometric", "linear", "flat_tail", "two_cluster", "one_spike"]:
        s = spectrum(kind, N, kappa)
        print(row(kind, measure(make_spd(s))))
    print()


def sweep_cond():
    N, kind = 1025, "geometric"
    print(f"[COND]   N={N}  shape={kind}  (how does the error grow with condition number?)")
    for kappa in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]:
        s = spectrum(kind, N, kappa)
        print(row(f"cond={kappa:.0e}", measure(make_spd(s))))
    print()


def sweep_size():
    kappa, kind = 1e6, "geometric"
    print(f"[SIZE]   shape={kind}  cond={kappa:.0e}  (does size / power-of-2 matter?)")
    for N in [64, 128, 256, 512, 1023, 1024, 1025, 2048, 4096]:
        s = spectrum(kind, N, kappa)
        print(row(f"N={N}", measure(make_spd(s))))
    print()


def sweep_gap():
    """Direct test of the relative-gap hypothesis: constant consecutive ratio r,
    i.e. geometric with cond = r^(N-1). Vary r; find the gap threshold."""
    N = 1025
    print(f"[GAP]    N={N}  geometric, constant consecutive ratio r  (s_i/s_i+1 = r)")
    for r in [1.001, 1.003, 1.005, 1.01, 1.02, 1.05, 1.1, 1.3]:
        kappa = r ** (N - 1)
        s = spectrum("geometric", N, kappa)
        print(row(f"r={r:<5}  (cond={kappa:.1e})", measure(make_spd(s))))
    print()


which = sys.argv[1] if len(sys.argv) > 1 else "shape"
if which in ("shape", "all"):
    sweep_shape()
if which in ("cond", "all"):
    sweep_cond()
if which in ("size", "all"):
    sweep_size()
if which in ("gap", "all"):
    sweep_gap()
