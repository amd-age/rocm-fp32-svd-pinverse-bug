# ROCm fp32 `torch.linalg.svd` / `torch.pinverse` accuracy bug (MI300X / gfx942)

Self-contained reproduction of an accuracy bug in **single-precision (fp32) SVD on
ROCm**. For an ill-conditioned symmetric positive-definite matrix, fp32
`torch.linalg.svd` / `torch.linalg.svdvals` (and therefore `torch.pinverse`,
which is built on the SVD) returns **grossly wrong small singular values** on an
AMD MI300X, while the *same call at the same precision on CPU* is accurate. fp64
is fine on both backends.

## TL;DR

| call | backend | result |
|---|---|---|
| fp32 `svdvals` / `svd` | **ROCm GPU (MI300X)** | mean singular-value error **43%**, 86% of values off by >10% |
| fp32 `svdvals` / `svd` | CPU | mean error 0.03% — accurate |
| fp32 `pinverse` (`A·pinv(A)`) | **ROCm GPU** | `‖A·pinv(A) − I‖/‖I‖ = 5.6` — **broken** |
| fp32 `pinverse` | CPU | `0.06` — at the expected fp32 conditioning floor |
| fp64 (svd / pinverse) | GPU or CPU | accurate |

It is **not** that fp32 is too low precision for this matrix — CPU fp32 handles it
fine. It is the **ROCm fp32 SVD path specifically**.

## Run it

```bash
python repro.py            # uses ./cxx_layer1.pt
```

Only dependency is `torch`. Run on AMD (ROCm) and on NVIDIA (CUDA) to compare;
on a correct backend every row is `OK`.

## The artifact

`cxx_layer1.pt` is a real `[1025, 1025]` fp32 symmetric positive-definite matrix
`A = XᵀX + reg·I`, condition number ≈ 8.8e5, full rank. `X` are key-projection
activations from a transformer layer (Llama-3.2-3B, layer 1 `k_proj`), and the
ridge term `reg·I` is the standard regularizer of a least-squares fit. The bug is
**structure-dependent** — a random matrix at the *same or higher* condition number
does not necessarily trigger it — so a representative real matrix is shipped
rather than generated. The matrix is the only input needed; no model or dataset
is required to reproduce.

## What the bug breaks in practice

This surfaced while computing ridge least-squares estimators `W = (XᵀX + reg·I)⁻¹·XᵀY`
via `pinverse`. On ROCm the corrupted pseudo-inverse silently produced wrong
regression weights — e.g. an identity fit `Y = X` (which must give ≈0 error)
returned relative MSE ≈ 0.06, i.e. negative explained variance. Switching the
solver to a Cholesky factorization (`torch.linalg.cholesky` + `torch.cholesky_solve`),
or to fp64, or running on CPU, all fix it. The defect is in the fp32 SVD, so any
code path that reaches `torch.linalg.svd`/`svdvals`/`pinverse` in fp32 on this
backend is affected.

## What structure triggers it

The trigger is **spectral shape, not condition number**: many singular values with
small relative gaps spread smoothly over a wide range (power-law / geometric decay,
cond ≳ 1e4) — exactly a real covariance + ridge matrix. It is **size-independent**
(an 8×8 matrix fails like a 4096×4096 one). A high cond from a single outlier
(random matrices) or from exactly-degenerate clusters does *not* trigger it. Full
characterization with sweeps in [`STRUCTURE.md`](STRUCTURE.md); regenerate with
`python structure_probe.py all`.

## Files

| file | what |
|---|---|
| `repro.py` | standalone reproduction on the real matrix (torch only) |
| `repro_synthetic.py` | model-free reproduction — generates a tiny matrix, no data file |
| `structure_probe.py` | sweeps spectrum shape / cond / size to characterize the trigger |
| `plot_size.py` / `svd_error_vs_size.png` | error-vs-size plot (CPU vs ROCm fp32) |
| `cxx_layer1.pt` | the offending real `[1025,1025]` fp32 SPD matrix (~4 MB) |
| `STRUCTURE.md` | what matrix structure triggers the bug (sweep results) |
| `ISSUE.md` | ready-to-file upstream bug report |

## Environment where observed

- AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, 304 CUs
- ROCm 7.1.0 / HIP 7.1.0, `torch 2.10.0.dev20251112+rocm7.1`
- Also reproduced on `torch 2.7.1+rocm7.2.2` (HIP 7.2) on the same GPU — i.e. it
  persists across a major torch/ROCm version change.
- `torch.backends.cuda.matmul.allow_tf32 = False` (default); the bug is unrelated
  to TF32.
- Fully deterministic: identical result on every repeat.
