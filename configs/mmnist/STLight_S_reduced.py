method = 'STLight'
# Reduced-width run that was actually trained (RTX 4060, 200 epochs).
# dim 424 was chosen when Table 3's 17.1M was misread as 3.3M.
# Measured: 3.34M params, 3.039G FLOPs. This is not Table 3 STLight-S.
# The Table 3 width is configs/mmnist/STLight_S.py (dim 1000, 17.1M).
dim = 424
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
