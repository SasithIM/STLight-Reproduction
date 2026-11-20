# Reproduction Notes & Decision Log

This file documents every significant decision made during the reproduction,
why it was made, and how it can be verified. Written to support the final
demonstration and any questions about the work.

## 1. The official code was never released — we transcribed it from the paper

**Decision:** reimplement STLight from the official implementation printed in
the paper's own supplementary appendix, instead of using the GitHub links.

**Evidence (verifiable):**
- `https://github.com/AlfaranoAndrea/STLight` — the link in the paper's footnote — is a
  plain OpenSTL fork: its 169 commits contain **zero** STLight files (checked full git
  history, `git log --all --name-only | grep -i stlight` is empty).
- `https://github.com/AlfaranoAndrea/STLight2` contains only a README saying
  *"The code will be soon publicly available!"*
- The full implementation exists as **Figure 10 in Appendix G** of the arXiv version
  ([arXiv:2411.10198](https://arxiv.org/abs/2411.10198), 16 pages; the WACV version is
  8 pages without the appendix). Appendix D (Listing 1) gives the weight initialization.

**What we did:** transcribed Figure 10 verbatim into `openstl/models/stlight_model.py`
and Listing 1 into its `_init_weights`. Our integration is annotated with the appendix
sources.

## 2. We integrated into the authors' own training framework

**Decision:** use the OpenSTL-Lightning fork (the repo the authors pointed to) as the
base, rather than building a standalone training script.

**Why:** the paper's experiments are run under OpenSTL ("we compare our results against
OpenSTL's public results, ensuring the same consistent training setting", Sec. 4.1).
Using the same framework means the data pipeline (Moving MNIST 10→10 frames, /255
normalization, fixed 10,000-sequence test set), the metrics (MSE/MAE/SSIM/PSNR on the
×255 scale), and the training recipe come from the exact protocol the paper used — so
our numbers are directly comparable to Table 3.

**Changes to the framework itself:** minimal and listed here in full —
- added `openstl/models/stlight_model.py`, `openstl/methods/stlight.py` (plain MSE
  objective, per the paper's "training objective is the Mean Squared Error"),
- registered `'stlight'` in both registries,
- `openstl/api/exp.py`: device auto-fallback (cuda → mps → cpu) so it also runs on
  Apple Silicon and laptops; `'stlight'` added to the FLOPs-measurement branch.
Nothing else was modified.

## 3. Model variant sizes were reverse-engineered from the paper's tables

**Decision:** STLight-XS = dim 224, depth 16; STLight-S = dim 424, depth 16.

**Why:** the paper reports results per variant (Table 3: XS 1.1M params / 1.0G FLOPs,
S 3.3M) but never publishes the per-variant configurations. What it does publish:
- Appendix B's parameter-count formula: repeated STLMixer cost ≈ `O(d·(d_ℓ·(a²+d+k₁²/3+k₂²/9)))`
- Table 8's anchor configuration for STLight-L: `dim=1400, depth=16` → our formula gives
  32.8M params, matching Table 3's 32.8M exactly, which validates the derivation.
- Solving for XS/S targets with K₁=3, K₂=7, patch=2 yields dim 224 (→ 1.05M params,
  ≈1.0G FLOPs measured) and dim 424 (→ 3.34M params).

**Measured vs paper:** XS 1.05M vs 1.1M (−4%); S 3.34M vs 3.3M (+1%). Reported honestly
in the results table.

## 4. Training recipe copied verbatim from the paper (Table 2)

Moving MNIST: Adam, lr=0.003, OneCycle scheduler with final_div_factor=1e4, batch size
16, 200 epochs, MSE loss. Kernel sizes K₁=3, K₂=7 follow Appendix C's kernel sweep
(Table 7); patch size 2 with no overlap (patch_size=2, overlapping=0). No data
augmentation, no tricks — same as the standardized OpenSTL setting.

## 5. Scope: one dataset, matching the assignment's "at least part of the results"

**Decision:** replicate the **Moving MNIST** benchmark only (not TaxiBJ/KTH/KITTI).

**Why:** MMNIST is the paper's primary benchmark (Fig. 1, Table 3 headline), it is
fully self-supervised (no external data), the test set is a standard fixed file (so
numbers are comparable), and it is the only dataset whose data pipeline needs no manual
download curation beyond two standard files. The guidelines require demonstrating the
code running with *at least part* of the results replicated.

## 6. Compute plan

- Primary: partner's RTX 4070 laptop GPU (~2–4 h for 200 epochs of XS).
- Backup: Kaggle notebook API (free P100/T4, 9–12 h sessions — one session suffices),
  scriptable via `kaggle kernels push/status/output`.
- Fallback: Apple M1 (MPS) — validated end-to-end with a 1-epoch smoke run
  (~8 min/epoch, too slow for the full schedule but proves the pipeline works).

## 7. What is (and is not) replicated

| | Status |
|---|---|
| Official architecture (App. G Fig. 10) | ✅ verbatim |
| Param/FLOPs counts vs Table 3 | ✅ XS 1.05M/≈1.0G (paper 1.1M/1.0G), S 3.34M (3.3M) |
| Training recipe (Table 2) | ✅ verbatim |
| MMNIST 200-epoch training + test | ✅ run (results table in README) |
| Qualitative predictions (Fig. 3 style) | ✅ via `tools/demo_gifs.py` / `tools/demo_predict.py` |
| TaxiBJ / KTH / KITTI results | ❌ out of scope (time budget) |
| Long-sequence (20-frame) prediction | ❌ out of scope |

## 8. Known variance sources (if our MSE differs from 24.48)

- Per-variant configs are our derivation (Sec. 3), not the authors'.
- GPU/framework/library version differences (PyTorch, Lightning) vs the paper's
  RTX A6000 setup; nondeterministic init and data shuffling.
- The paper itself trains each variant with tuned hidden dims per dataset (Sec. 4.5),
  and its own appendix (Table 8) shows MSE moving by ~5 points across plausible
  hyperparameter ranges — so a few points of deviation on MMNIST is within the
  reproduction noise band for this class of model.
