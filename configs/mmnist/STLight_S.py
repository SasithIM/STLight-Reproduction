method = 'STLight'
# Table 3 STLight-S on Moving MNIST: 17.1M params, 16.5G FLOPs.
# dim=1000, depth=16 gives 17.09M parameters. Not yet trained.
# The completed 4060 run used configs/mmnist/STLight_S_reduced.py (dim 424).
dim = 1000
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
