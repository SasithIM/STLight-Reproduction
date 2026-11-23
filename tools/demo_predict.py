"""Live demo: load a trained STLight checkpoint and predict the next 10 frames
of Moving MNIST sequences in seconds (CPU/MPS/GPU), producing a Figure-3-style
GIF and per-sample MSE. Use this to demonstrate the code running.

Usage (from repo root):
    python tools/demo_predict.py --exp_dir results/mmnist_stlight_xs --n_samples 3
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


def find_checkpoint(exp_dir):
    candidates = [
        osp.join(exp_dir, 'checkpoints', 'best.ckpt'),
        osp.join(exp_dir, 'checkpoints', 'last.ckpt'),
        osp.join(exp_dir, 'best.ckpt'),
        osp.join(exp_dir, 'last.ckpt'),
    ]
    found = next((p for p in candidates if osp.exists(p)), None)
    if found is None:
        raise FileNotFoundError(f'no checkpoint under {exp_dir}')
    return found


def load_model(exp_dir, device):
    ckpt_path = find_checkpoint(exp_dir)
    blob = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = blob['state_dict']
    state = {k.replace('model.', '', 1): v for k, v in state.items() if k.startswith('model.')}
    cfg_path = osp.join(exp_dir, 'model_param.json')
    if osp.exists(cfg_path):
        config = json.load(open(cfg_path))
    else:
        # results/ stores best.ckpt at the top level, without model_param.json.
        net_ids = [int(k.split('.')[1]) for k in state
                   if k.startswith('net.') and k.split('.')[1].isdigit()]
        config = {
            'in_shape': [10, 1, 64, 64],
            'aft_seq_length': 10,
            'dim': int(state['patch_encoder.0.weight'].shape[0]),
            'depth': max(net_ids) + 1,
            'patch_size': 2,
            'overlapping': 0,
            'K_1': 3,
            'K_2': 7,
        }
    in_shape = config['in_shape']  # [T_in, C, H, W]
    model = STLight_Model(in_shape=in_shape,
                          aft_seq_length=config['aft_seq_length'],
                          dim=config.get('dim', 224),
                          depth=config.get('depth', 16),
                          patch_size=config.get('patch_size', 2),
                          overlapping=config.get('overlapping', 0),
                          K_1=config.get('K_1', 3),
                          K_2=config.get('K_2', 7))
    model.load_state_dict(state)
    config['ckpt_path'] = ckpt_path
    return model.to(device).eval(), config


def get_test_samples(data_path, n_samples, seed=0):
    seqs = np.load(data_path, mmap_mode='r')
    if seqs.shape[0] == 20:  # Official Moving MNIST stores time before sample.
        seqs = seqs.transpose(1, 0, 2, 3)
    rng = np.random.RandomState(seed)
    idx = rng.choice(seqs.shape[0], size=n_samples, replace=False)
    batch = seqs[idx, :10].astype(np.float32) / 255.0   # first 10 frames = input
    target = seqs[idx, 10:20].astype(np.float32) / 255.0  # last 10 frames = target
    return torch.from_numpy(batch)[:, :, None], target  # (N,10,64,64) -> (N,10,1,64,64)


def save_filmstrip(rows, titles, out_path):
    """Column t is frame t. rows are (n_frames, H, W)."""
    n_rows, n_frames = len(rows), rows[0].shape[0]
    fig, axes = plt.subplots(n_rows, n_frames, figsize=(1.15 * n_frames, 1.25 * n_rows))
    axes = np.atleast_2d(axes)
    for ax_row, row, title in zip(axes, rows, titles):
        ax_row[0].set_ylabel(title, rotation=90, fontsize=8, labelpad=6)
        for t, ax in enumerate(ax_row):
            ax.imshow(row[t], cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if ax_row is axes[0]:
                ax.set_title(str(t + 1), fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_playback(rows, titles, out_path, fps=4):
    n_frames = rows[0].shape[0]
    fig, axes = plt.subplots(1, len(rows), figsize=(2.3 * len(rows), 2.6))
    axes = np.atleast_1d(axes)
    imgs = []
    for ax, row, title in zip(axes, rows, titles):
        imgs.append(ax.imshow(row[0], cmap='gray', vmin=0, vmax=1))
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()

    def update(t):
        for img, row in zip(imgs, rows):
            img.set_data(row[t])
        return imgs

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='data/moving_mnist/mnist_test_seq.npy')
    parser.add_argument('--n_samples', type=int, default=3)
    parser.add_argument('--scan', type=int, default=0,
                        help='score this many random sequences and keep the n_samples nearest the median MSE')
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
    print(f'checkpoint: {config["ckpt_path"]}')
    print(f'STLight ({config.get("dim")}x{config.get("depth")}) — {n_params/1e6:.2f}M params')

    n_draw = args.scan if args.scan and args.scan > args.n_samples else args.n_samples
    x, target = get_test_samples(args.data_path, n_draw)
    with torch.no_grad():
        pred = model(x.to(device)).cpu().numpy()
    target_np = target
    # Table 3 MSE is OpenSTL's: mean over time, then sum over H×W×C.
    # pred is (N, T, C, H, W); target_np is (N, T, H, W).
    spatial = pred.shape[-2] * pred.shape[-1] * pred.shape[2]
    mses = [float(np.mean((pred[i, :, 0] - target_np[i]) ** 2)) * spatial
            for i in range(pred.shape[0])]
    if n_draw != args.n_samples:
        mid = float(np.median(mses))
        keep = sorted(range(len(mses)), key=lambda i: abs(mses[i] - mid))[:args.n_samples]
        print(f'scanned {n_draw} sequences, median OpenSTL MSE = {mid:.2f} (Table 3 scale)')
    else:
        keep = list(range(pred.shape[0]))

    out_dir = osp.join(args.exp_dir, 'demo')
    import os
    os.makedirs(out_dir, exist_ok=True)
    titles = ['input (given)', 'ground truth', 'STLight prediction']
    for out_i, i in enumerate(keep):
        rows = [x[i, :, 0].numpy(), target_np[i], pred[i, :, 0]]
        save_filmstrip(rows, titles, osp.join(out_dir, f'filmstrip_{out_i}.png'))
        save_playback(rows, titles, osp.join(out_dir, f'playback_{out_i}.gif'))
        print(f'sample {out_i}: OpenSTL MSE = {mses[i]:.2f} -> filmstrip_{out_i}.png, playback_{out_i}.gif')
    print(f'\nArtifacts saved to {out_dir}')


if __name__ == '__main__':
    main()
