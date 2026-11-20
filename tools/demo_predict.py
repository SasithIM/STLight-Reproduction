"""Live demo: load a trained STLight checkpoint and predict the next 10 frames
of Moving MNIST sequences in seconds (CPU/MPS/GPU), producing a Figure-3-style
GIF and per-sample MSE. Use this to demonstrate the code running.

Usage (from repo root):
    python tools/demo_predict.py --exp_dir work_dirs/mmnist_stlight_xs --n_samples 3
"""
import argparse
import json
import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import torch

from openstl.models.stlight_model import STLight_Model


def load_model(exp_dir, device):
    with open(osp.join(exp_dir, 'model_param.json')) as f:
        config = json.load(f)
    in_shape = config['in_shape']  # [T_in, C, H, W]
    model = STLight_Model(in_shape=in_shape,
                          aft_seq_length=config['aft_seq_length'],
                          dim=config.get('dim', 224),
                          depth=config.get('depth', 16),
                          patch_size=config.get('patch_size', 2),
                          overlapping=config.get('overlapping', 0),
                          K_1=config.get('K_1', 3),
                          K_2=config.get('K_2', 7))
    ckpts = [osp.join(exp_dir, 'checkpoints', 'best.ckpt'),
             osp.join(exp_dir, 'checkpoints', 'last.ckpt')]
    ckpt_path = next(p for p in ckpts if osp.exists(p))
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
    state = {k.replace('model.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval(), config


def get_test_samples(data_path, n_samples, seed=0):
    seqs = np.load(data_path)  # (10000, 20, 64, 64) uint8, 20-frame sequences
    rng = np.random.RandomState(seed)
    idx = rng.choice(seqs.shape[0], size=n_samples, replace=False)
    batch = seqs[idx, :10].astype(np.float32) / 255.0   # first 10 frames = input
    target = seqs[idx, 10:20].astype(np.float32) / 255.0  # last 10 frames = target
    return torch.from_numpy(batch)[:, :, None], target  # (N,10,64,64) -> (N,10,1,64,64)


def make_gif(rows, titles, out_path, fps=4):
    n_rows, n_frames = len(rows), rows[0].shape[0]
    fig, axes = plt.subplots(n_rows, n_frames,
                             figsize=(1.6 * n_frames, 1.6 * n_rows))
    axes = np.atleast_2d(axes)
    for ax_row, title in zip(axes, titles):
        ax_row[0].set_ylabel(title, rotation=90, fontsize=9, labelpad=8)
    imgs = []
    for ax_row, row in zip(axes, rows):
        imgs.append([ax.imshow(row[0], cmap='gray', vmin=0, vmax=1) for ax in ax_row])
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([])

    def update(t):
        return [img.set_data(row[t]) for row_imgs, row in zip(imgs, rows)
                for img in row_imgs]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='data/moving_mnist/mnist_test_seq.npy')
    parser.add_argument('--n_samples', type=int, default=3)
    parser.add_argument('--device', type=str, default=None,
                        help='cuda / mps / cpu (default: auto)')
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'device: {device}')

    model, config = load_model(args.exp_dir, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'STLight ({config.get("dim")}x{config.get("depth")}) — {n_params/1e6:.2f}M params')

    x, target = get_test_samples(args.data_path, args.n_samples)
    with torch.no_grad():
        pred = model(x.to(device)).cpu().numpy()
    target_np = target

    out_dir = osp.join(args.exp_dir, 'demo')
    import os
    os.makedirs(out_dir, exist_ok=True)
    for i in range(pred.shape[0]):
        mse = float(np.mean((pred[i] - target_np[i]) ** 2)) * 255 ** 2
        inp = x[i, :, 0].numpy()
        make_gif([inp, target_np[i, :, 0], pred[i, :, 0]],
                 ['input (given)', 'ground truth', 'STLight prediction'],
                 osp.join(out_dir, f'live_sample_{i}.gif'))
        print(f'sample {i}: MSE(x255) = {mse:.2f} -> live_sample_{i}.gif')
    print(f'\nGIFs saved to {out_dir}')


if __name__ == '__main__':
    main()
