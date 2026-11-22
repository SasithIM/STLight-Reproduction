"""Generate demo artifacts (Figure-3 style GIFs + a metrics report) from saved
test outputs of an OpenSTL experiment.

Usage (from repo root, after running exp.test()):
    venv/bin/python tools/demo_gifs.py --res_dir work_dirs/mmnist_stlight_xs

Creates, inside <res_dir>/demo/:
    sample_<i>.gif  - input | ground truth | prediction, side by side
    error_<i>.gif   - |target - prediction| heatmap
    summary.txt     - dataset/sample info and per-sample MSE
"""
import argparse
import os.path as osp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def make_gif(rows, titles, out_path, fps=4, figsize=(2.2, 2.2)):
    """rows: list of (n_frames, H, W) arrays rendered as separate grid rows."""
    n_rows, n_frames = len(rows), rows[0].shape[0]
    fig, axes = plt.subplots(n_rows, n_frames, figsize=(figsize[0] * n_frames, figsize[1] * n_rows))
    axes = np.atleast_2d(axes)
    imgs = [[ax.imshow(row[0], cmap='gray', vmin=0, vmax=1) for ax in ax_row]
            for ax_row, row in zip(axes, rows)]
    for ax_row, title in zip(axes, titles):
        ax_row[0].set_ylabel(title, rotation=90, fontsize=9, labelpad=8)
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([])

    def update(t):
        for row_imgs, row in zip(imgs, rows):
            for img in row_imgs:
                img.set_data(row[t])
        return [img for row_imgs in imgs for img in row_imgs]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', type=str, required=True,
                        help='experiment dir containing saved/{inputs,preds,trues}.npy')
    parser.add_argument('--n_samples', type=int, default=3)
    args = parser.parse_args()

    saved = osp.join(args.res_dir, 'saved')
    inputs = np.load(osp.join(saved, 'inputs.npy'), mmap_mode='r')
    preds = np.load(osp.join(saved, 'preds.npy'), mmap_mode='r')
    trues = np.load(osp.join(saved, 'trues.npy'), mmap_mode='r')
    metrics = np.load(osp.join(saved, 'metrics.npy')) if osp.exists(osp.join(saved, 'metrics.npy')) else None

    demo_dir = osp.join(args.res_dir, 'demo')
    import os
    os.makedirs(demo_dir, exist_ok=True)

    # overall metrics
    squared_error = absolute_error = 0.0
    for start in range(0, len(preds), 64):
        diff = preds[start:start + 64] - trues[start:start + 64]
        squared_error += float(np.sum(diff ** 2, dtype=np.float64))
        absolute_error += float(np.sum(np.abs(diff), dtype=np.float64))
    mse, mae = squared_error / preds.size, absolute_error / preds.size
    lines = [f'samples: {preds.shape[0]}  in-frames: {inputs.shape[1]}  out-frames: {preds.shape[1]}',
             f'MSE (this test set): {mse:.2f}   MAE: {mae:.2f}',
             'paper Table 3 (MMNIST, 200 ep): XS MSE 24.48 / MAE 71.21 ; S MSE 23.29 / MAE 68.33',
             '']
    if metrics is not None:
        lines.append(f'metrics.npy (mae, mse): {metrics.tolist()}')

    # per-sample GIFs
    rng = np.random.RandomState(0)
    idxs = rng.choice(preds.shape[0], size=min(args.n_samples, preds.shape[0]), replace=False)
    for i in idxs:
        inp = inputs[i, :, 0]
        tgt, pred = trues[i, :, 0], preds[i, :, 0]
        err = np.abs(tgt - pred)
        gif_path = osp.join(demo_dir, f'sample_{i}.gif')
        make_gif([inp, tgt, pred], ['input', 'target', 'STLight pred'], gif_path)
        make_gif([err], ['|target - pred|'], osp.join(demo_dir, f'error_{i}.gif'))
        per_sample_mse = float(np.mean((pred - tgt) ** 2))
        lines.append(f'sample {i}: per-sample MSE {per_sample_mse:.2f} -> {osp.basename(gif_path)}')

    with open(osp.join(demo_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nArtifacts written to {demo_dir}')


if __name__ == '__main__':
    main()
