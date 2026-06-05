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

Only dependency is `torch` (the plots/sweeps additionally use `numpy` +
`matplotlib`). Run on AMD (ROCm) and on NVIDIA (CUDA) to compare; on a correct
backend every row is `OK`.

### On the shipped real matrix

```bash
python repro.py                  # uses ./cxx_layer1.pt
```

### On synthesized matrices (no data file needed)

The bug reproduces on synthetic SPD matrices with a geometric singular-value
spectrum, so you can trigger it without the shipped `.pt`. The matrix is built as
`A = Q diag(s) Qᵀ` with `s` a geometric spectrum and `Q` Haar-random orthogonal
(see `repro_synthetic.py` / `STRUCTURE.md`).

```bash
python repro_synthetic.py            # default: N=16, cond=1e6
python repro_synthetic.py 8   1e6    # tiny: an 8x8 matrix already fails
python repro_synthetic.py 4096 1e6   # large; size does not matter
python repro_synthetic.py 512 1e4    # onset of the failure (cond ~1e4)
python repro_synthetic.py 512 1e3    # below onset: OK even on ROCm
```

`repro_synthetic.py [N] [cond]` prints CPU-vs-ROCm fp32 singular-value error and a
verdict. To map the failure across spectral shape / condition number / size and
regenerate the plots:

```bash
python structure_probe.py all        # shape, cond, size, gap sweeps -> tables
python plot_size.py                  # svd_error_vs_size{,_fp64}.png  (geometric spectrum)
python plot_random_size.py           # svd_error_vs_size_random.png   (RANDOM singular values)
python plot_cond.py                  # svd_error_vs_cond.png          (error vs cond)
```

`plot_random_size.py` draws the singular values at random (log-uniform across the
range) instead of on a geometric law — it triggers the bug identically, confirming
it's not an artifact of the exact spectrum. Its verdict is **relative to CPU fp32**
(the achievable fp32 baseline), not an absolute threshold: at very high condition
number fp32 cannot resolve the small singular values on *any* backend, so the bug
is defined as ROCm doing far worse than CPU *while CPU is still accurate*.

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
| `plot_size.py` / `svd_error_vs_size*.png` | error-vs-size plots, fp32 + fp64 control (geometric) |
| `plot_random_size.py` / `svd_error_vs_size_random.png` | error-vs-size with RANDOM singular values |
| `random_matrix_probe.py` | random-ENTRY matrices (no QSQᵀ); shows the bug is not universal |
| `plot_normal_size.py` / `svd_error_vs_size_normal.png` | control: error-vs-size for ordinary Gaussian matrices (ROCm fine) |
| `plot_baseline_size.py` / `svd_error_vs_size_fixedcond.png` | error-vs-size at FIXED cond — isolates ROCm's ~N^1.2 dimension effect |
| `plot_cond.py` / `svd_error_vs_cond.png` | error-vs-condition-number, geometric spectrum (CPU vs ROCm) |
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
