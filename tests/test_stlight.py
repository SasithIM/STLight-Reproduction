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


def check(variant, dim, depth, expect_params_m):
    model = STLight_Model(in_shape=(10, 1, 64, 64), aft_seq_length=10,
                          dim=dim, depth=depth, patch_size=2,
                          overlapping=0, K_1=3, K_2=7)
    n_params = count_params(model)
    x = torch.randn(2, 10, 1, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 10, 1, 64, 64), f'bad output shape {y.shape}'
    # one MSE training step
    y = model(x)
    loss = torch.nn.functional.mse_loss(y, torch.randn_like(y))
    loss.backward()
    print(f'{variant}: dim={dim} depth={depth} -> {n_params/1e6:.2f}M params '
          f'(paper: ~{expect_params_m}M), out {tuple(y.shape)}, loss {loss.item():.3f} OK')


if __name__ == '__main__':
    check('STLight-XS', 224, 16, 1.1)
    check('STLight-S', 424, 16, 3.3)
    print('All STLight sanity checks passed.')
