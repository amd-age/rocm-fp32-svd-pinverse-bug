"""
Model-free reproduction: NO data file needed. Generates a tiny SPD matrix with a
smooth (geometric) spectrum and shows that fp32 `torch.linalg.svdvals` on ROCm
returns wrong singular values, while CPU fp32 (same input) is accurate.

The bug is size-independent: an 8x8 matrix already triggers it. The trigger is the
SPECTRAL SHAPE -- many singular values with small relative gaps over a wide range
(condition number alone is not predictive; see STRUCTURE.md).

    python repro_synthetic.py [N] [cond]      # defaults: N=16, cond=1e6

Only dependency: torch.
"""
import sys
import torch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
cond = float(sys.argv[2]) if len(sys.argv) > 2 else 1e6

print(f"torch={torch.__version__} hip={getattr(torch.version,'hip',None)} "
      f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
print(f"synthetic SPD matrix: N={N}  cond={cond:.0e}  spectrum=geometric (smooth decay)\n")

# A = Q diag(s) Qᵀ, Q Haar-random orthogonal, s geometric from 1 to 1/cond. All fp64.
g = torch.Generator().manual_seed(0)
Q, _ = torch.linalg.qr(torch.randn(N, N, dtype=torch.float64, generator=g))
s = (1.0 / cond) ** (torch.arange(N, dtype=torch.float64) / (N - 1))
A = (Q * s) @ Q.T
A = ((A + A.T) / 2).to(torch.float32)             # the fp32 matrix both backends receive

ref = torch.linalg.svdvals(A.double())             # fp64 SVD of that exact fp32 matrix = truth
print(f"  {'backend':>7} | {'mean σ rel-err':>14} {'max σ rel-err':>14} {'frac σ >10% off':>16}")
means = {}
for dev in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
    S = torch.linalg.svdvals(A.to(dev)).double().cpu()
    rel = (S - ref).abs() / ref.abs()
    means[dev] = rel.mean().item()
    print(f"  {dev:>7} | {means[dev]:>14.3e} {rel.max().item():>14.3e} {(rel>0.1).float().mean().item():>16.2f}")

# Verdict relative to the achievable fp32 baseline (CPU), NOT an absolute threshold.
# At high cond fp32 cannot resolve the small singular values on ANY backend (the
# error is floored at ~eps_fp32*cond by Weyl's theorem; CPU fp32 hits that floor
# too). So we read CPU fp32 as "what fp32 can achieve here":
#   - CPU still accurate but ROCm large+worse  -> backend bug (fp32 COULD have worked)
#   - CPU itself large                          -> fp32 precision floor, not hardware
if "cuda" in means:
    cpu_m, rocm_m = means["cpu"], means["cuda"]
    ratio = rocm_m / cpu_m if cpu_m > 0 else float("inf")
    if cpu_m > 0.1:
        print(f"\n  VERDICT: fp32 precision floor exceeded on BOTH backends "
              f"(cond too high for fp32; CPU err={cpu_m:.2f}). Not hardware-specific. "
              f"ROCm {ratio:.0f}x worse.")
    elif rocm_m > 1e-3 and ratio > 5:
        print(f"\n  VERDICT: *** ROCm BROKEN *** -- {ratio:.0f}x worse than CPU fp32, "
              f"while CPU fp32 is accurate (err={cpu_m:.1e}). fp32 could resolve this; "
              f"ROCm does not.")
    elif ratio > 5:
        print(f"\n  VERDICT: ROCm {ratio:.0f}x worse than CPU but both small (benign baseline)")
    else:
        print(f"\n  VERDICT: OK -- ROCm matches CPU fp32 (matrix easy for fp32)")
