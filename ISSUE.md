# fp32 `torch.linalg.svd` / `svdvals` / `pinverse` returns wrong small singular values on ROCm (MI300X / gfx942)

## 🐛 Describe the bug

On an AMD Instinct MI300X (gfx942), single-precision (fp32) `torch.linalg.svdvals`,
`torch.linalg.svd`, and `torch.pinverse` return **grossly inaccurate small
singular values** for an ill-conditioned (cond ≈ 9e5) symmetric positive-definite
matrix. The largest singular value is correct, but ~86% of the singular values are
off by more than 10% (mean relative error **43%**, max **102%**).

The **same calls at the same fp32 precision on the CPU are accurate** (mean error
0.03%), and **fp64 is accurate on both GPU and CPU**. So this is not a precision
limitation of fp32 for this matrix — it is specific to the ROCm fp32 SVD path
(presumably the hipSOLVER/rocSOLVER `gesvd` backend that `linalg.svd` dispatches
to). Because `torch.pinverse` is built on the SVD, it returns a badly wrong
pseudo-inverse: for a full-rank SPD matrix `A`, `‖A·pinv(A) − I‖_F / ‖I‖_F = 5.6`
on ROCm fp32 vs `0.06` on CPU fp32 (the latter being the expected
conditioning-limited floor, ≈ cond·ε_fp32).

`svdvals` (singular values only, no singular vectors) fails identically, which
localizes the fault to the singular-value computation itself rather than the
formation of U/V.

This is fully deterministic and reproduces across torch/ROCm versions (observed on
both `2.10.0.dev20251112+rocm7.1` and `2.7.1+rocm7.2.2`).

## To Reproduce

A self-contained repro (one ~4 MB matrix + a short script, `torch` only) is
attached. The matrix `A` is a real `[1025, 1025]` fp32 SPD matrix `XᵀX + reg·I`
(cond ≈ 8.8e5). The defect is structure-dependent, so a representative real matrix
is shipped rather than generated randomly.

```python
import torch

A = torch.load("cxx_layer1.pt")                 # [1025,1025] fp32, SPD, cond ~8.8e5
S_ref = torch.linalg.svdvals(A.double())        # fp64 reference spectrum

for dev in ("cpu", "cuda"):
    S = torch.linalg.svdvals(A.to(dev, torch.float32)).double().cpu()
    rel = (S - S_ref).abs() / S_ref.abs()
    print(f"{dev:>4} fp32 svdvals: mean rel-err={rel.mean():.3e}  "
          f"max={rel.max():.3e}  frac>10%={(rel>0.1).float().mean():.2f}")

    A_d = A.to(dev, torch.float32)
    I = torch.eye(A.shape[0], device=dev)
    inv_res = ((A_d @ torch.pinverse(A_d) - I).norm() / I.norm()).item()
    print(f"{dev:>4} fp32 pinverse: ||A pinv(A) - I||/||I|| = {inv_res:.3e}")
```

### Observed output (MI300X, ROCm 7.1, torch 2.10.0.dev+rocm7.1)

```
 cpu fp32 svdvals: mean rel-err=8.195e-05  max=1.243e-03  frac>10%=0.00
 cpu fp32 pinverse: ||A pinv(A) - I||/||I|| = 6.258e-02
cuda fp32 svdvals: mean rel-err=4.331e-01  max=1.016e+00  frac>10%=0.86
cuda fp32 pinverse: ||A pinv(A) - I||/||I|| = 5.589e+00
```

Full table from `repro.py`:

```
torch.pinverse quality (A is SPD full-rank, so pinv(A) should equal inv(A)):
  backend  dtype | ||A·pinv(A)-I||/||I|| | ||A·pinv·A-A||/||A|| | verdict
     cuda   fp32 |          5.588942e+00 |         2.603213e-02 | *** BROKEN ***
      cpu   fp32 |          6.060736e-02 |         2.518054e-03 | OK (at fp32 floor)
     cuda   fp64 |          1.336236e-05 |         5.355239e-10 | OK
      cpu   fp64 |          2.133311e-11 |         8.184171e-11 | OK

torch.linalg.svd fp32, singular values vs fp64 reference:
  backend | recon relerr | max σ relerr | mean σ relerr | frac σ >10% off
      cpu |    1.136e-05  |    1.471e-02 |     2.635e-03 |           0.00
     cuda |    1.410e-03  |    1.016e+00 |     4.331e-01 |           0.86

smallest 5 singular values:
  fp64 ref : [2.2377e+02, 2.2376e+02, 2.2373e+02, 2.2371e+02, 2.2369e+02]
  cpu  fp32: [2.2380e+02, 2.2375e+02, 2.2373e+02, 2.2369e+02, 2.2367e+02]
  cuda fp32: [3.0518e+02, 2.9260e+02, 2.9264e+02, 2.9747e+02, 3.0679e+02]  <- wrong
```

The SVD reconstruction error `‖U diag(S) Vᵀ − A‖/‖A‖` is `1.4e-3` on ROCm fp32 vs
`1.1e-5` on CPU fp32 (~100×), confirming the decomposition itself is wrong, not
just an ordering/labeling difference.

## Expected behavior

ROCm fp32 `svdvals`/`svd`/`pinverse` should match CPU fp32 to within normal
floating-point tolerance (mean singular-value relative error ~1e-3, not ~4e-1).
On a correct backend every row above is `OK`.

## Versions

```
PyTorch version: 2.10.0.dev20251112+rocm7.1
HIP runtime version: 7.1.0
GPU: AMD Instinct MI300X (gfx942:sramecc+:xnack-), 304 CUs
ROCm: 7.1.0
torch.backends.cuda.matmul.allow_tf32: False
```

Also reproduced on PyTorch `2.7.1+rocm7.2.2` (HIP 7.2) on the same GPU.

## Additional context

- **Workaround:** for the ridge least-squares use case that surfaced this, replacing
  `pinverse` with a Cholesky solve (`torch.linalg.cholesky` + `torch.cholesky_solve`,
  valid since `A` is SPD) is numerically stable and faster. fp64 SVD or running the
  SVD on CPU also work. But fp32 `linalg.svd`/`svdvals` on ROCm being this
  inaccurate is the underlying bug.
- The matrix is well within fp32's representable range (entries O(1e5), no overflow)
  and is numerically symmetric and positive definite.
- Likely lives in the rocSOLVER/hipSOLVER `gesvd` path used by `linalg_svd` on ROCm;
  a `gesvdj`-style iterative refinement or a tighter convergence tolerance for
  clustered small singular values may be the fix.
