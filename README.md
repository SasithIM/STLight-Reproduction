# STLight — Course Reproduction (WACV 2025)

Course project reproducing **"STLight: a Fully Convolutional Approach for Efficient
Predictive Learning by Spatio-Temporal joint Processing"** (Alfarano et al., WACV 2025,
[arXiv:2411.10198](https://arxiv.org/abs/2411.10198)).

> **Where is the code from?** The authors' GitHub repos ([STLight](https://github.com/AlfaranoAndrea/STLight),
> [STLight2](https://github.com/AlfaranoAndrea/STLight2)) are **empty placeholders** — the promised
> OpenSTL integration was never released. However, the **full official implementation is printed in
> the arXiv paper's supplementary appendix** (Appendix G, Figure 10, plus the weight-initialization
> scheme of Appendix D, Listing 1). This repository transcribes that implementation verbatim and
> integrates it into the OpenSTL training framework the authors used.

## Repository layout

This repo is a fork of [chengtan9907/OpenSTL](https://github.com/chengtan9907/OpenSTL)
(`OpenSTL-Lightning` branch, v1.0.0, PyTorch Lightning) — the same base the paper builds on.
On top of it we added:

| File | Purpose |
|---|---|
| `openstl/models/stlight_model.py` | **The official STLight model** (from arXiv Fig. 10): spatio-temporal patch encoder, `STLMixer` blocks (depthwise K_1 → dilated depthwise K_2 → pointwise), skip connection from `depth/3` to `2·depth/3`, PixelShuffle + 1×1-conv decoder, Kaiming-Normal init (Listing 1) |
| `openstl/methods/stlight.py` | OpenSTL method wrapper (plain MSE training objective, as in the paper) |
| `configs/mmnist/STLight_XS.py` | Table 3 STLight-XS: dim=800, depth=16 → **11.11M params** (paper: 11.1M / 10.6G). Not trained yet |
| `configs/mmnist/STLight_S.py` | Table 3 STLight-S: dim=1000, depth=16 → **17.09M params** (paper: 17.1M / 16.5G). Not trained yet |
| `configs/mmnist/STLight_XS_reduced.py` | Width that was trained: dim=224 → **1.05M / 0.87G**. Came from reading 11.1M as 1.1M |
| `configs/mmnist/STLight_S_reduced.py` | Width that was trained: dim=424 → **3.34M / 3.04G**. Came from reading 17.1M as 3.3M |
| `tests/test_stlight.py` | Sanity checks: param counts, output shape, one training step |
| `tools/demo_gifs.py` | Generates Figure-3-style comparison GIFs (input/target/prediction) from test outputs |

Both configs use the paper's Table 2 Moving-MNIST training recipe:
**Adam, lr=0.003, OneCycle (final_div_factor=1e4), batch 16, 200 epochs, MSE loss**.

## Setup

Requires **Python 3.10** (OpenSTL constraint: ≤3.10.8).

```bash
# 1. Environment
python3.10 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install --upgrade pip

# torch: default wheels bundle CUDA on Windows/Linux; on Apple Silicon they bundle MPS
pip install torch torchvision

# 2. OpenSTL-Lightning runtime deps
pip install "lightning==2.2.1" einops fvcore hickle lpips matplotlib "numpy<2" \
    opencv-python pandas scikit-image scikit-learn "timm==0.9.2" tqdm xarray dask \
    netcdf4 packaging "setuptools<81" six future PyWavelets xlsxwriter tensorboard

# NOTE: timm must be 0.9.2 (newer versions renamed the optimizer modules OpenSTL imports)
# NOTE: setuptools must be <81 (newer versions removed pkg_resources, needed by lightning 2.2.1)

# 3. Install the openstl package (editable, no deps re-resolution)
pip install -e . --no-deps

# 4. Verify the model
python tests/test_stlight.py
```

## Data — Moving MNIST

```bash
mkdir -p data/moving_mnist
cd data/moving_mnist
# train-set source images (generated sequences each epoch)
curl -LO https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
# standard fixed test set (10000 sequences, 20 frames) — needed for paper-comparable MSE
curl -LO http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
cd ../..
```

## Training

```bash
# Table 3 widths (not trained yet). XS is about 13x the completed reduced run.
python tools/train.py -d mmnist -c configs/mmnist/STLight_XS.py --ex_name mmnist_stlight_xs_table3
python tools/train.py -d mmnist -c configs/mmnist/STLight_S.py --ex_name mmnist_stlight_s_table3

# Completed 4060 runs (reduced width). Checkpoints are in results/.
python tools/train.py -d mmnist -c configs/mmnist/STLight_XS_reduced.py --ex_name mmnist_stlight_xs
python tools/train.py -d mmnist -c configs/mmnist/STLight_S_reduced.py --ex_name mmnist_stlight_s
```

- Epochs come from the config (`epoch = 200`). For quick tests override via a copy of the
  config or edit it (`epoch = 1` → ~10 min on a laptop GPU).
- Checkpoints land in `work_dirs/<ex_name>/checkpoints/` (`best.ckpt` + `last.ckpt`).
- The console prints the model summary and an fvcore **FLOPs table** at startup —
  capture that for the demo (params/FLOPs vs. paper).

## Testing + demo artifacts

```bash
# runs the test set with the best checkpoint and saves inputs/preds/trues/metrics
python tools/train.py -d mmnist -c configs/mmnist/STLight_XS.py --ex_name mmnist_stlight_xs --test

# Figure-3-style strips from a finished test directory (needs saved/preds.npy)
python tools/demo_gifs.py --res_dir work_dirs/mmnist_stlight_xs --n_samples 3

# Live demo from the committed checkpoints (no work_dirs layout required)
python tools/demo_predict.py --exp_dir results/mmnist_stlight_xs --n_samples 3
python tools/demo_predict.py --exp_dir results/mmnist_stlight_s --n_samples 3
```

## Results (Moving MNIST 10→10)

Paper numbers are Table 3 of the WACV / arXiv paper (MMNIST columns). Ours are the
RTX 4060 Laptop runs committed in `results/`, 200 epochs, Table 2 recipe, seed 42.
Test metrics are the OpenSTL end-of-training test (last-epoch weights, errors on the
0–255 scale). Curves are in `results/training_curves.png`.

| Model | MSE ↓ | MAE ↓ | SSIM ↑ | Params | FLOPs |
|---|---|---|---|---|---|
| STLight-XS (paper Table 3) | 24.48 | 71.21 | 0.9444 | 11.1M | 10.6G |
| STLight-S (paper Table 3) | 23.29 | 68.33 | 0.9454 | 17.1M | 16.5G |
| STLight-L (paper Table 3) | 21.80 | 66.92 | 0.9515 | 32.9M | 32.3G |
| Ours, reduced XS (dim 224) | 40.00 | 102.29 | 0.8924 | 1.05M | 0.87G |
| Ours, reduced S (dim 424) | 30.39 | 87.00 | 0.9156 | 3.34M | 3.04G |

The completed runs are the right architecture and the right training recipe, at the
wrong width. Table 3 prints **11.1M / 10.6G** for XS and **17.1M / 16.5G** for S.
Those were first read as 1.1M / 1.0G and 3.3M. The widths that actually hit the
table are dim 800 (11.11M) and dim 1000 (17.09M); dim 1400 reproduces the appendix
STLight-L anchor at 32.88M. Scaling the measured FLOPs by those widths lands next
to 10.6G and 16.5G, so the block itself matches and only the channel count was short.

What the reduced runs still show:

- S beats XS on every metric, which is the paper's ordering.
- Both loss curves descend for 200 epochs (S val loss 0.031 → 0.0074, no spikes).
- XS val loss spiked above 0.05 on 9 epochs while the learning rate was high, then recovered. Best val is epoch 189 (0.00954); the reported test is the last epoch (val 0.00977).
- Wall time on the 4060: XS about 8 h 50 min, S about 17 h 50 min.
- Qualitative predictions are in `results/*/demo/`. The original GIFs repeat one frame across every column. `filmstrip_*.png` / `playback_*.gif` are corrected time strips from `best.ckpt` (three sequences near the median of 64 test clips). `tools/demo_predict.py` prints per-clip error on the same OpenSTL scale as Table 3.

A Table 3 match needs the dim-800 and dim-1000 configs. Those are roughly 6–13× the
compute of the finished runs, so they were not started on this machine.

## Final demonstration

`slides/STLight_demonstration.pptx` is the demonstration deck, with speaker notes. `DEMO.md` has the architecture, the comparison, why the width is smaller, and the live command.

Live command, from the repo root with the virtualenv active:

```bash
python tools/demo_predict.py --exp_dir results/mmnist_stlight_s --n_samples 3 --scan 64
```

The fair comparison is the trained S model (3.34M, 3.0G, MSE 30.39) against the published same-size rows: SimVP+ConvMixer (3.9M, 5.5G, MSE 32.09) and PhyDNet (3.1M, 15.3G, MSE 28.19). Trained XS is the smaller internal check. Paper STLight-XS and STLight-S are the full-width targets, and they were not trained here.

## Attribution

- Paper: Alfarano A., Alfarano A., Friso L., Bacciu A., Amerini I., Silvestri F. — WACV 2025.
- Model implementation transcribed from the paper's own Appendix G (Apache-2.0 OpenSTL base,
  upstream README preserved at `README_OpenSTL_upstream.md`).
