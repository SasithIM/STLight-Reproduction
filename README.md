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
| `configs/mmnist/STLight_XS.py` | STLight-XS: dim=224, depth=16 → **1.05M params / ~1.0G FLOPs** (paper Table 3: 1.1M / 1.0G) |
| `configs/mmnist/STLight_S.py` | STLight-S: dim=424, depth=16 → **3.34M params** (paper: 3.3M) |
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
# STLight-XS (paper Table 3 MMNIST: MSE 24.48, MAE 71.21, SSIM 0.9444, 1.1M params, 1.0G FLOPs)
python tools/train.py -d mmnist -c configs/mmnist/STLight_XS.py --ex_name mmnist_stlight_xs

# STLight-S (paper: MSE 23.29, MAE 68.33, SSIM 0.9454, 3.3M params)
python tools/train.py -d mmnist -c configs/mmnist/STLight_S.py --ex_name mmnist_stlight_s
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

# Figure-3-style GIFs + metrics summary
python tools/demo_gifs.py --res_dir work_dirs/mmnist_stlight_xs --n_samples 3
```

## Expected results (paper Table 3, Moving MNIST 10→10)

| Model | MSE ↓ | MAE ↓ | SSIM ↑ | Params | FLOPs |
|---|---|---|---|---|---|
| STLight-XS (paper) | 24.48 | 71.21 | 0.9444 | 1.1M | 1.0G |
| STLight-S (paper) | 23.29 | 68.33 | 0.9454 | 3.3M | 16.5G |
| ConvLSTM (paper) | 28.80 | 90.04 | 0.9258 | 15.0M | 36.6G |
| PredRNN (paper) | 23.66 | 69.98 | 0.9462 | 3.8M | 17.1G |

(Our measured values will be filled in after the 200-epoch runs complete.)

## Attribution

- Paper: Alfarano A., Alfarano A., Friso L., Bacciu A., Amerini I., Silvestri F. — WACV 2025.
- Model implementation transcribed from the paper's own Appendix G (Apache-2.0 OpenSTL base,
  upstream README preserved at `README_OpenSTL_upstream.md`).
