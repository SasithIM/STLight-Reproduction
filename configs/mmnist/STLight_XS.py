method = 'STLight'
# Table 3 STLight-XS on Moving MNIST: 11.1M params, 10.6G FLOPs.
# dim=800, depth=16 gives 11.11M parameters. Not yet trained.
# The completed 4060 run used configs/mmnist/STLight_XS_reduced.py (dim 224).
dim = 800
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
