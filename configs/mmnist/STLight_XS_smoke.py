method = 'STLight'
# model (STLight-XS)
dim = 224
depth = 16
patch_size = 2
overlapping = 0
K_1 = 3
K_2 = 7
# training — 1-epoch SMOKE TEST config: verifies the pipeline end to end
# (train + validation + test + artifacts) in ~10 minutes before the real run
lr = 0.003
batch_size = 16
sched = 'onecycle'
final_div_factor = 1e4
opt = 'adam'
epoch = 1
