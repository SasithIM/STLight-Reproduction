method = 'STLight'
# model (STLight-S: ~3.3M params as in Table 3)
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
