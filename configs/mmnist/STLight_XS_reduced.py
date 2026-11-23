method = 'STLight'
# Reduced-width run that was actually trained (RTX 4060, 200 epochs).
# dim 224 was chosen when Table 3's 11.1M was misread as 1.1M.
# Measured: 1.05M params, 0.871G FLOPs. This is not Table 3 STLight-XS.
# The Table 3 width is configs/mmnist/STLight_XS.py (dim 800, 11.1M).
dim = 224
depth = 16
patch_size = 2
overlapping = 0
K_1 = 3
K_2 = 7
# training (Table 2, MMNIST)
lr = 0.003
batch_size = 16
sched = 'onecycle'
final_div_factor = 1e4
opt = 'adam'
epoch = 200
