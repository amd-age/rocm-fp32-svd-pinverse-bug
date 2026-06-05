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
print(f"  {'backend':>7} | {'mean σ rel-err':>14} {'max σ rel-err':>14} {'frac σ >10% off':>16}  verdict")
for dev in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
    S = torch.linalg.svdvals(A.to(dev)).double().cpu()
    rel = (S - ref).abs() / ref.abs()
    mean, mx, frac = rel.mean().item(), rel.max().item(), (rel > 0.1).float().mean().item()
    verdict = "*** BROKEN ***" if mean > 1e-2 else "OK"
    print(f"  {dev:>7} | {mean:>14.3e} {mx:>14.3e} {frac:>16.2f}  {verdict}")
