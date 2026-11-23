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

## 3. Model variant sizes — first derivation was a misread, then corrected

**What was trained:** STLight reduced-XS = dim 224, depth 16 (1.05M / 0.87G);
reduced-S = dim 424, depth 16 (3.34M / 3.04G). Those are the 4060 checkpoints.

**Why those widths:** Table 3 was read as XS 1.1M / 1.0G and S 3.3M. The printed
MMNIST cells are **11.1M / 10.6G** and **17.1M / 16.5G** (confirmed on the rendered
WACV page and the arXiv text layer). The leading 1 of 11.1M was dropped.

**Correct widths, same formula.** Table 8's STLight-L anchor `dim=1400, depth=16`
gives 32.88M, matching Table 3's 32.9M. With K₁=3, K₂=7, patch=2, depth=16:

| Variant | dim | Params | Paper |
|---|---|---|---|
| XS | 800 | 11.11M | 11.1M / 10.6G |
| S | 1000 | 17.09M | 17.1M / 16.5G |
| M | 1200 | 24.34M | 24.3M / 23.7G |
| L | 1400 | 32.88M | 32.9M / 32.3G |

These are in `configs/mmnist/STLight_XS.py` and `STLight_S.py`. The trained widths
are kept as `STLight_XS_reduced.py` and `STLight_S_reduced.py`. The Table 3 widths
have not been trained. FLOPs measured on the reduced models, scaled by `(dim_paper/dim_trained)²`,
land within a few percent of 10.6G and 16.5G, which checks the block rather than the width.

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
| Param/FLOPs counts vs Table 3 | ❌ trained 1.05M/0.87G and 3.34M/3.04G. Table 3 is 11.1M/10.6G and 17.1M/16.5G. Correct dims (800, 1000) checked, not trained |
| Training recipe (Table 2) | ✅ verbatim |
| MMNIST 200-epoch training + test | ✅ reduced widths only (README table). Table 3 widths not trained |
| Qualitative predictions (Fig. 3 style) | ✅ filmstrips from `best.ckpt` via `tools/demo_predict.py` |
| TaxiBJ / KTH / KITTI results | ❌ out of scope (time budget) |
| Long-sequence (20-frame) prediction | ❌ out of scope |

## 8. Why our MSE is 40.0 / 30.4 rather than 24.48 / 23.29

The gap is the width error in Sec. 3, not a failed training run. The reduced XS
model has about one tenth of Table 3 XS's parameters (1.05M vs 11.1M) and the
reduced S model about one fifth (3.34M vs 17.1M). Both still train: S is stable
for all 200 epochs; XS has nine validation spikes above 0.05 while the OneCycle
learning rate is high, then finishes at val loss 0.0098. The test numbers below
are the last-epoch OpenSTL test. `best.ckpt` is the lowest validation checkpoint
(epoch 189 for XS, epoch 199 for S).

| | MSE | MAE | SSIM | PSNR |
|---|---|---|---|---|
| Reduced XS, last epoch | 39.999 | 102.287 | 0.8924 | 21.005 |
| Reduced S, last epoch | 30.392 | 87.001 | 0.9156 | 22.181 |
| Paper XS / S | 24.48 / 23.29 | 71.21 / 68.33 | 0.9444 / 0.9454 | — |

Smaller leftover differences, once the width is corrected: GPU and library versions
versus the paper's RTX A6000, and Table 8's own finding that MSE moves by several
points across nearby learning rates.
