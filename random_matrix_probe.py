"""
Does the ROCm fp32 svdvals bug need the QSQᵀ construction? No. Here we build
matrices from RANDOM ENTRIES (no prescribed spectrum, no SVD construction) and use
the fp64 SVD of the same fp32 matrix as the reference -- which we already showed is
trustworthy. This also shows the bug is NOT universal: well-conditioned random
matrices are fine; only ill-conditioned-AND-dense spectra trigger it.

Families (all random entries):
  gaussian   : A = randn(N, N)                         -> well-conditioned (cond ~ N)
  wishart    : A = G Gᵀ / M, G = randn(N, M), M>N      -> SPD, moderate cond
  scaled     : A = randn(N, N) · diag(geom 1..1/c)     -> random entries, ill-conditioned
  corr_cov   : A = (X·diag(geom)) (·)ᵀ + small ridge   -> random "data" covariance, dense decay

Reference = fp64 SVD of the fp32 matrix. Verdict is relative to CPU fp32 (the
achievable fp32 baseline), not an absolute threshold.

    python random_matrix_probe.py
Only dependency: torch.
"""
import torch

N = 1024
C = 1e6            # target dynamic range for the ill-conditioned families
SEEDS = range(3)
DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
print(f"torch={torch.__version__} hip={getattr(torch.version,'hip',None)} "
      f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}  N={N}\n")


def make(kind, gen):
    if kind == "gaussian":
        return torch.randn(N, N, dtype=torch.float64, generator=gen)
    if kind == "wishart":
        M = N + N // 4
        G = torch.randn(N, M, dtype=torch.float64, generator=gen)
        return (G @ G.T) / M
    if kind == "scaled":
        A = torch.randn(N, N, dtype=torch.float64, generator=gen)
        d = (1.0 / C) ** (torch.arange(N, dtype=torch.float64) / (N - 1))   # column scaling
        return A * d
    if kind == "corr_cov":
        # random "data" with correlated, geometrically-weighted factors -> dense decaying spectrum
        Msamp = 4 * N
        F = torch.randn(Msamp, N, dtype=torch.float64, generator=gen)
        w = (1.0 / C) ** (torch.arange(N, dtype=torch.float64) / (N - 1))
        X = F * w                                  # scale feature columns
        return (X.T @ X) / Msamp + (1e-7) * torch.eye(N, dtype=torch.float64)
    raise ValueError(kind)


def verdict(cpu_m, rocm_m):
    r = rocm_m / cpu_m if cpu_m > 0 else float("inf")
    if cpu_m > 0.1:
        return f"fp32 floor on BOTH (not hardware); ROCm {r:.0f}x worse"
    if rocm_m > 1e-3 and r > 5:
        return f"*** ROCm BROKEN *** {r:.0f}x worse than CPU fp32"
    if r > 5:
        return f"ROCm {r:.0f}x worse but both small (benign)"
    return "ROCm ~ CPU (fine)"


print(f"  {'family':>9} | {'cond(A)':>9} | {'cpu fp32':>9} {'rocm fp32':>9} | verdict")
for kind in ["gaussian", "wishart", "scaled", "corr_cov"]:
    conds, cpu_es, rocm_es = [], [], []
    for sd in SEEDS:
        g = torch.Generator().manual_seed(sd)
        A64 = make(kind, g)
        A32 = A64.to(torch.float32)
        ref = torch.linalg.svdvals(A32.double())          # fp64 reference, no QSQᵀ needed
        conds.append((ref[0] / ref[-1]).item())
        for dev in DEVS:
            S = torch.linalg.svdvals(A32.to(dev)).double().cpu()
            e = ((S - ref).abs() / ref.abs()).mean().item()
            (cpu_es if dev == "cpu" else rocm_es).append(e)
    cond = sum(conds) / len(conds)
    cpu_m = sum(cpu_es) / len(cpu_es)
    rocm_m = (sum(rocm_es) / len(rocm_es)) if rocm_es else float("nan")
    print(f"  {kind:>9} | {cond:>9.1e} | {cpu_m:>9.2e} {rocm_m:>9.2e} | {verdict(cpu_m, rocm_m)}")
