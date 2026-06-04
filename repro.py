"""
Minimal reproduction: fp32 torch.linalg.svd / torch.pinverse is inaccurate on
ROCm (AMD MI300X, gfx942) for an ill-conditioned symmetric positive-definite
matrix. CPU fp32 and any fp64 path are fine; only the ROCm fp32 SVD path fails.

Ships one artifact: cxx_layer1.pt -- a real [1025, 1025] fp32 SPD matrix
A = XᵀX + reg·I with condition number ~8.8e5 (provenance in README.md). The bug
is structure-dependent, so a representative matrix is shipped rather than
generated randomly.

Usage:
    python repro.py                 # uses ./cxx_layer1.pt
    python repro.py path/to/A.pt

Only dependency: torch. Run on AMD (ROCm) and NVIDIA (CUDA) and compare.
Expected on a correct backend: every row is OK.
Observed on ROCm: the GPU/fp32 rows are BROKEN.
"""
import sys
import torch

HAS_GPU = torch.cuda.is_available()
print(f"torch={torch.__version__}  hip={getattr(torch.version, 'hip', None)}  "
      f"cuda={torch.version.cuda}  "
      f"gpu={torch.cuda.get_device_name(0) if HAS_GPU else 'cpu-only'}")
print(f"allow_tf32(matmul)={torch.backends.cuda.matmul.allow_tf32}  "
      f"allow_tf32(cudnn)={torch.backends.cudnn.allow_tf32}\n")

path = sys.argv[1] if len(sys.argv) > 1 else "cxx_layer1.pt"
A_cpu64 = torch.load(path, map_location="cpu").double()          # [N, N] SPD
N = A_cpu64.shape[0]
S_ref = torch.linalg.svdvals(A_cpu64)                            # fp64 reference spectrum
cond = (S_ref[0] / S_ref[-1]).item()
print(f"A: shape={tuple(A_cpu64.shape)}  symmetric={'yes' if torch.allclose(A_cpu64, A_cpu64.T) else 'no'}  "
      f"cond(A)={cond:.3e}  (full rank, SPD)\n")

I_ref = torch.eye(N, dtype=torch.float64)


def pinv_quality(where, dtype):
    """Measure how close pinverse(A) is to a true inverse (A is full-rank SPD)."""
    A = A_cpu64.to(where, dtype)
    P = torch.pinverse(A).double().cpu()
    a = A_cpu64
    inv_res = ((a @ P - I_ref).norm() / I_ref.norm()).item()        # ||A·pinv(A) - I||
    mp_res = (((a @ P @ a) - a).norm() / a.norm()).item()           # ||A·pinv(A)·A - A||
    return inv_res, mp_res


def svd_quality(where):
    """fp32 SVD on `where`, singular values compared to the fp64 reference."""
    A = A_cpu64.to(where, torch.float32)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    S = S.double().cpu()
    recon = ((U.double().cpu() @ torch.diag(S) @ Vh.double().cpu()) - A_cpu64).norm().item() / A_cpu64.norm().item()
    rel = (S - S_ref).abs() / S_ref.abs()
    return recon, rel.max().item(), rel.mean().item(), (rel > 0.1).float().mean().item(), S


# --- pinverse / inverse quality -------------------------------------------
# The inverse residual ||A·pinv(A) - I|| is bounded below by the conditioning:
# an accurate fp32 solve still floors at ~cond(A)·eps_fp32. CPU fp32 sits at that
# floor (its SVD is accurate); a backend whose residual is orders of magnitude
# above the floor has computed a wrong decomposition.
EPS32 = torch.finfo(torch.float32).eps
fp32_floor = cond * EPS32
print(f"expected fp32 inverse-residual floor ~ cond·eps_fp32 = {fp32_floor:.3e}\n")
print("torch.pinverse quality  (A is SPD full-rank, so pinv(A) should equal inv(A)):")
print(f"  {'backend':>7} {'dtype':>6} | {'||A·pinv(A)-I||/||I||':>22} {'||A·pinv·A-A||/||A||':>22}  verdict")
configs = [("cpu", torch.float32), ("cpu", torch.float64)]
if HAS_GPU:
    configs = [("cuda", torch.float32), ("cpu", torch.float32),
               ("cuda", torch.float64), ("cpu", torch.float64)]
for where, dtype in configs:
    inv_res, mp_res = pinv_quality(where, dtype)
    if dtype == torch.float64:
        verdict = "OK" if inv_res < 1e-2 else "*** BROKEN ***"
    else:   # fp32: OK if within ~10x of the conditioning-limited floor
        verdict = "OK" if inv_res < 10 * fp32_floor else "*** BROKEN ***"
    print(f"  {where:>7} {('fp32' if dtype == torch.float32 else 'fp64'):>6} | "
          f"{inv_res:>22.6e} {mp_res:>22.6e}  {verdict}")

# --- SVD (the primitive pinverse is built on) -----------------------------
print("\ntorch.linalg.svd  fp32, singular values vs fp64 reference  (same precision, different backend):")
print(f"  {'backend':>7} | {'recon relerr':>13} {'max σ relerr':>13} {'mean σ relerr':>14} {'frac σ >10% off':>16}")
S_cpu = S_gpu = None
recon, mx, mn, frac, S_cpu = svd_quality("cpu")
print(f"  {'cpu':>7} | {recon:>13.3e} {mx:>13.3e} {mn:>14.3e} {frac:>16.2f}")
if HAS_GPU:
    recon, mx, mn, frac, S_gpu = svd_quality("cuda")
    print(f"  {'cuda':>7} | {recon:>13.3e} {mx:>13.3e} {mn:>14.3e} {frac:>16.2f}")
    print(f"\n  smallest 5 singular values:")
    print(f"    fp64 ref : {[f'{v:.4e}' for v in S_ref[-5:].tolist()]}")
    print(f"    cpu  fp32: {[f'{v:.4e}' for v in S_cpu[-5:].tolist()]}")
    print(f"    cuda fp32: {[f'{v:.4e}' for v in S_gpu[-5:].tolist()]}   <- ROCm fails to resolve these")
