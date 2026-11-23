# Partner Run Guide — RTX 4070 Laptop

**Status (after the 4060 runs):** both 200-epoch jobs finished and are in `results/`.
The widths in this guide (dim 224 and 424) are **not** Table 3. The paper's XS/S
cells are 11.1M and 17.1M parameters; see `DECISIONS.md` §3 and the README results
table before quoting 1.05M / 3.34M as the paper's models.

**Your job in one sentence:** get the two 200-epoch training runs done on your 4070 and
record the artifacts listed in §6 — everything else is already prepared.

This reproduces *STLight (WACV 2025)* on Moving MNIST. The model code is the paper's own
(verbatim from its appendix — see `DECISIONS.md` §1); the training recipe is the paper's
Table 2. Your runs produce the numbers we compare against Table 3 in the demo.

**Total time budget: ~30 min setup + overnight training.**

---

## 1. Before you start (5 min)

- [ ] Ask Sasith to add your GitHub username as a collaborator on
      `SasithIM/STLight-Reproduction` (Settings → Collaborators).
- [ ] Install **Python 3.10** (exact version matters — not 3.11+):
      - Windows: https://www.python.org/downloads/release/python-31011/ → "Windows installer (64-bit)",
        tick **"Add python.exe to PATH"** during install.
      - Check: `py -3.10 --version`
- [ ] Have ~4 GB free disk (data + checkpoints) and the laptop **charger within reach**.

## 2. Clone + environment (~15 min)

Open PowerShell / terminal in a workspace folder:

```powershell
git clone https://github.com/SasithIM/STLight-Reproduction.git
cd STLight-Reproduction

py -3.10 -m venv venv
.\venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)

python -m pip install --upgrade pip

# torch (Windows wheel bundles CUDA automatically)
pip install torch torchvision

# framework deps — the two PINS are mandatory, do not "upgrade" them:
pip install "lightning==2.2.1" einops fvcore hickle lpips matplotlib "numpy<2" `
    opencv-python pandas scikit-image scikit-learn "timm==0.9.2" tqdm xarray dask `
    netcdf4 packaging "setuptools<81" six future PyWavelets xlsxwriter tensorboard

pip install -e . --no-deps
```

> **Why the pins?** timm >0.9 renamed the optimizer modules OpenSTL imports; setuptools ≥81
> removed `pkg_resources` which lightning 2.2.1 needs. Wrong versions = immediate ImportError.

## 3. Verify the model (2 min)

```powershell
python tests/test_stlight.py
```

Expected output — **screenshot this, we use it in the demo**:

```
STLight-XS: dim=224 depth=16 -> 1.05M params (paper: ~1.1M), out (2, 10, 1, 64, 64), loss ... OK
STLight-S: dim=424 depth=16 -> 3.34M params (paper: ~3.3M), out (2, 10, 1, 64, 64), loss ... OK
All STLight sanity checks passed.
```

## 4. Data (~10 min, one-time)

```powershell
mkdir data\moving_mnist
cd data\moving_mnist
curl.exe -L -o train-images-idx3-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl.exe -L -o mnist_test_seq.npy "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
cd ..\..
```

(On Windows use `curl.exe`, not `curl` — the latter aliases a slow PowerShell cmdlet.
The second file is ~800 MB; browser download works too, just save it into
`data/moving_mnist/`. Verify sizes: ~9.9 MB and ~819 MB.)

## 5. Training runs

### 5a. Smoke test first (~10 min) — do not skip

```powershell
python tools/train.py -d mmnist -c configs/mmnist/STLight_XS_smoke.py --ex_name smoke_xs --val_batch_size 64
```

It runs 1 full epoch + validation + test (~5–10 min on the 4070). If it finishes and
prints a metrics block with MSE/MAE/SSIM, **the whole pipeline works on your machine**.
The final block (metrics table) is artifact **R4** (§6) — copy it even from the smoke run.

### 5b. The real run — STLight-XS (~4–8 h)

```powershell
python tools/train.py -d mmnist -c configs/mmnist/STLight_XS.py --ex_name mmnist_stlight_xs --val_batch_size 64
```

Notes:
- The console shows a **model summary + FLOPs table** right after startup — that's
  artifact **R1**, copy it immediately (Ctrl+C on the terminal text / screenshot).
- Epochs auto-run to 200 (from the config). Expect roughly 1.5–3 min/epoch; note the
  actual time for the first epoch — we report it in the demo.
- **Keep the laptop plugged in and set power/sleep to "never" for the session.**
- Progress bar shows `train_loss_step` — it should drop fast in the first ~20 epochs
  (ours hit ~0.16 by step 33 in the smoke run). If loss goes to NaN or explodes,
  stop, screenshot, message us.

### 5c. Run 2 — STLight-S (start after XS finishes, or on the second night)

```powershell
python tools/train.py -d mmnist -c configs/mmnist/STLight_S.py --ex_name mmnist_stlight_s --val_batch_size 64
```

Same procedure. S is ~3× bigger — proportionally longer.

### If the run gets interrupted

`work_dirs/<ex_name>/checkpoints/last.ckpt` is saved every epoch — tell us and we'll
sort out resuming. Don't delete the folder.

## 6. What to RECORD (the deliverables back to the team)

For **each** of the two real runs (`mmnist_stlight_xs`, `mmnist_stlight_s`):

| # | Artifact | Where it comes from | How |
|---|---|---|---|
| R1 | **Params + FLOPs table** | printed right after training starts | copy-paste terminal text into a `.txt`, or screenshot |
| R2 | **Training log file** | `work_dirs/<ex_name>/train_*.log` | auto-saved — just keep it |
| R3 | **Training curve CSV** | `work_dirs/<ex_name>/metrics/version_0/metrics.csv` | auto-saved — just keep it (we plot it) |
| R4 | **Final test metrics block** | printed at the end of the run (`mse`, `mae`, `ssim`, `psnr`) | copy-paste text + screenshot |
| R5 | **Comparison GIFs** | see command below | files |
| R6 | **Live-predict demo GIFs + timing** | see command below | files + screenshot of the terminal output |
| R7 | **Screen recording (insurance)** | optional but recommended: 2–3 min capture of R6 running | OBS / Xbox Game Bar (Win+Alt+R) |

R5 — after the run finishes:

```powershell
python tools/demo_gifs.py --res_dir work_dirs/mmnist_stlight_xs --n_samples 3
```

R6 — the "code running live" moment (seconds, CPU is fine):

```powershell
python tools/demo_predict.py --exp_dir work_dirs/mmnist_stlight_xs --n_samples 3
```

The terminal prints `device: cuda`, the param count, and per-sample MSE — **screenshot
that terminal** and keep the `live_sample_*.gif` files it writes.

## 7. Send everything back (5 min)

Copy the artifacts into the repo and push (checkpoints are small enough for GitHub):

```powershell
mkdir results\mmnist_stlight_xs
copy work_dirs\mmnist_stlight_xs\train_*.log results\mmnist_stlight_xs\
copy work_dirs\mmnist_stlight_xs\metrics\version_0\metrics.csv results\mmnist_stlight_xs\
copy work_dirs\mmnist_stlight_xs\saved\metrics.npy results\mmnist_stlight_xs\
copy work_dirs\mmnist_stlight_xs\demo\*.* results\mmnist_stlight_xs\
copy work_dirs\mmnist_stlight_xs\checkpoints\best.ckpt results\mmnist_stlight_xs\
# (repeat for mmnist_stlight_s, plus your R1/R4 text files saved into the same folder)

git pull
git add results
git commit -m "XS + S run artifacts from 4070"
git push
```

(If `best.ckpt` exceeds GitHub's 100 MB limit — unlikely, ~4–13 MB each — upload it to
Google Drive instead and share the link.)

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: timm.optim.nadam` | you installed a newer timm → `pip install "timm==0.9.2"` |
| `No module named 'pkg_resources'` | `pip install "setuptools<81"` |
| `FileNotFoundError ... train-images-idx3-ubyte.gz` | data files missing → redo §4, check you're in repo root when training |
| `CUDA out of memory` | shouldn't happen (model is tiny); if it does: `--batch_size 8` |
| Windows DataLoader hangs at start | kill and rerun; if persistent add `--num_workers 0` (slower) |
| Loss = NaN | stop, screenshot the console, message us |

**Any error not covered here: screenshot the full traceback and send it — do not
"fix" dependency versions on your own** (the pins are deliberate).

## 9. TL;DR checklist

1. ☐ Collaborator invite → clone → venv → pinned installs → `pip install -e . --no-deps`
2. ☐ `python tests/test_stlight.py` → screenshot (1.05M / 3.34M params)
3. ☐ Download the 2 data files into `data/moving_mnist/`
4. ☐ Smoke run (1 epoch) → keep its metrics block
5. ☐ Real run XS (200 ep) → **R1 FLOPs table** at startup
6. ☐ Real run S (200 ep) → **R1** again
7. ☐ R2–R6 for both runs (mostly auto-saved + two demo commands)
8. ☐ Copy into `results/`, `git push`
