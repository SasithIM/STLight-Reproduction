"""Sanity checks for the STLight integration.

Verifies (1) parameter counts against Table 3 of the paper, (2) output shape,
(3) that one training step (MSE backward) works.

Run from the repo root:  venv/bin/python tests/test_stlight.py
"""
import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

import torch

from openstl.models.stlight_model import STLight_Model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def check(variant, dim, depth, expect_params_m, tol=0.15):
    model = STLight_Model(in_shape=(10, 1, 64, 64), aft_seq_length=10,
                          dim=dim, depth=depth, patch_size=2,
                          overlapping=0, K_1=3, K_2=7)
    n_params = count_params(model)
    x = torch.randn(2, 10, 1, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 10, 1, 64, 64), f'bad output shape {y.shape}'
    got = n_params / 1e6
    assert abs(got - expect_params_m) < tol, (
        f'{variant} has {got:.2f}M params, expected ~{expect_params_m}M')
    # one MSE training step
    y = model(x)
    loss = torch.nn.functional.mse_loss(y, torch.randn_like(y))
    loss.backward()
    print(f'{variant}: dim={dim} depth={depth} -> {got:.2f}M params '
          f'(target: ~{expect_params_m}M), out {tuple(y.shape)}, loss {loss.item():.3f} OK')


if __name__ == '__main__':
    # Table 3 widths. L is the appendix anchor: dim=1400, depth=16 -> 32.9M.
    check('Table3 STLight-XS', 800, 16, 11.1)
    check('Table3 STLight-S', 1000, 16, 17.1)
    check('Table3 STLight-L', 1400, 16, 32.9)
    # Widths of the completed 4060 checkpoints (Table 3 was misread as 1.1M / 3.3M).
    check('Trained reduced XS', 224, 16, 1.05)
    check('Trained reduced S', 424, 16, 3.34)
    print('All STLight sanity checks passed.')
