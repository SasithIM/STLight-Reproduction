# Final demonstration

This is the note for the demonstration. The deck is `slides/STLight_demonstration.pptx`. Speaker notes are inside that file.

## What to run

From the repo root, with the project virtualenv active and `data/moving_mnist/mnist_test_seq.npy` present:

```bash
python tools/demo_predict.py --exp_dir results/mmnist_stlight_s --n_samples 3 --scan 64
```

The script is `tools/demo_predict.py`. It loads `results/mmnist_stlight_s/best.ckpt`, draws 64 random Moving MNIST test sequences, and keeps the 3 nearest the median error. It prints the device, `STLight (424x16)`, the parameter count, and an OpenSTL MSE for each clip. It writes:

- `results/mmnist_stlight_s/demo/filmstrip_0.png` (and `_1`, `_2`): columns are time. Top row is the 10 given frames, middle is the true future, bottom is the prediction.
- `results/mmnist_stlight_s/demo/playback_0.gif` (and `_1`, `_2`): the same three rows, animated.

A single clip will not print 30.39. That number is the mean over all 10,000 test sequences.

The same command with `--exp_dir results/mmnist_stlight_xs` runs the smaller model.

## Results already on disk

Both models finished 200 epochs on an RTX 4060 Laptop. The full-test numbers are the OpenSTL scale used in the paper's Table 3 (mean over batch and time, then sum over the 64×64 pixels). They come from the end-of-training test, which uses the last-epoch weights.

| Model | MSE | MAE | SSIM | Params | FLOPs |
| --- | --- | --- | --- | --- | --- |
| Ours, trained S (`dim` 424) | 30.39 | 87.00 | 0.916 | 3.34M | 3.0G |
| SimVP+ConvMixer (published) | 32.09 | 88.93 | 0.926 | 3.9M | 5.5G |
| PhyDNet (published) | 28.19 | 78.64 | 0.937 | 3.1M | 15.3G |
| Ours, trained XS (`dim` 224) | 40.00 | 102.29 | 0.892 | 1.05M | 0.87G |
| Paper STLight-S (`dim` 1000, not trained) | 23.29 | 68.33 | 0.945 | 17.1M | 16.5G |
| Paper STLight-XS (`dim` 800, not trained) | 24.48 | 71.21 | 0.944 | 11.1M | 10.6G |

Lower MSE and MAE is better. Higher SSIM is better. The SimVP and PhyDNet rows are the authors' published Table 3 runs, not a second run on our GPU.

Files to show if the live script cannot be run:

- `results/training_curves.png` — both 200-epoch curves.
- `results/mmnist_stlight_s/demo/filmstrip_0.png` — one median test clip of the trained S model. `filmstrip_1.png` and `filmstrip_2.png` are the other two.
- `results/mmnist_stlight_s/demo/playback_0.gif` — the same clip in motion.
- The XS copies are under `results/mmnist_stlight_xs/demo/`.

`results/*/demo/summary.txt` prints MSE 0.01. That is per-pixel error on pixels scaled to [0, 1]. It is not the Table 3 number. The Table 3 number for the trained S model is 30.39, also stored in that folder's `metrics.npy`.

The older `sample_*.gif` and `live_sample_*.gif` files repeat one frame across every column. Use the filmstrips and the playback GIFs.

The test sequence file is about 781MB and is gitignored. The checkpoints, filmstrips, curves, and deck are in the repo.

## What `dim` is

`dim` is the width of the hidden representation: how many channels each patch has after the first convolution.

The 10 input frames are stacked into 10 channels. The patch convolution maps those 10 channels to `dim` channels, and every one of the 16 STLMixer blocks stays at that width. A depthwise layer uses one filter per channel. The pointwise 1×1 layer mixes all `dim` channels with each other, so it has about `dim × dim` weights. The model size grows with `dim` squared. Depth stays 16.

PixelShuffle then splits those channels back into pixels, and the last 1×1 convolution turns them into the 10 output frames.

| Run | `dim` | Parameters |
| --- | --- | --- |
| Trained XS | 224 | 1.05M |
| Trained S | 424 | 3.34M |
| Paper STLight-XS | 800 | 11.1M |
| Paper STLight-S | 1000 | 17.1M |

## Our architecture

STLight takes 10 frames of 64×64 and returns the next 10 in one pass. Nothing is fed back in. The loss is mean squared error. The implementation is `openstl/models/stlight_model.py`, transcribed from Appendix G, Figure 10, with the Appendix D initialisation.

1. **Stack.** The 10 frames are laid on the channel axis, so the tensor is 10×64×64. Time is present before the first layer.
2. **Patch.** One convolution with stride 2, then batch norm and GELU, makes 32×32 patches with `dim` channels. The trained S model uses 424.
3. **Mix, 16 times.** Each STLMixer block is three convolutions. A depthwise 3×3 looks at nearby patches. A depthwise 7×7 with dilation 3 looks farther away. Those two sit in a residual branch, followed by GELU and batch norm. A pointwise 1×1 then mixes the channels. Because the channels began as frames, this is where time is mixed.
4. **Skip.** The representation after block 5 is added back before block 10, so a later block still sees an earlier state.
5. **Decode.** PixelShuffle turns each 32×32 patch back into 64×64 pixels. One 1×1 convolution turns the channels into 10 output frames.

## How the comparison models differ

The trained S model is this network at `dim` 424. SimVP+ConvMixer and PhyDNet are different networks of about the same size. Trained XS is the same STLight with fewer channels. Paper STLight-S is the same STLight with more channels, and it was not trained here.

### SimVP+ConvMixer

This is also one forward pass, and it also ends in convolutions. The split of the work is different. The OpenSTL config is `configs/mmnist/simvp/SimVP_ConvMixer.py`.

SimVP encodes each frame by itself. Its encoder is four convolutions with 64 channels, and it downsamples 64×64 to 32×32. Time is still a batch dimension there, so the first layers never see two frames together. A skip from the first encoder layer is added near the end of the decoder.

Only the middle block, the translator, stacks the 10 encoded frames on the channel axis. That translator is 8 ConvMixer blocks, widened to 512 channels. One ConvMixer block is a single depthwise 11×11, with a residual, then a pointwise 1×1. The decoder upsamples back to 64×64 and a 1×1 convolution writes each frame.

| | Trained STLight-S | SimVP+ConvMixer |
| --- | --- | --- |
| When time enters | First convolution | Only the middle translator |
| Spatial encoder | One stride-2 convolution | Four convolutions, with a decoder skip |
| Mixer | 16 blocks: depthwise 3×3, dilated depthwise 7×7, pointwise 1×1 | 8 blocks: one depthwise 11×11, then pointwise 1×1 |
| Decoder | PixelShuffle and one 1×1 | Several upsampling convolutions |
| Size | 3.34M, 3.0G | 3.9M, 5.5G |
| MSE | 30.39, our run | 32.09, the authors' run |

The paper describes STLMixer as a ConvMixer with two depthwise stages, a small kernel and a dilated one, instead of one large kernel. Our MSE is a little lower and our FLOPs are lower. Their SSIM is a little higher, 0.926 against 0.916.

### PhyDNet

PhyDNet is recurrent, and it splits the prediction into two latent branches. The code is `openstl/models/phydnet_model.py`.

A small CNN encodes one frame into a 16×16 latent. PhyCell is one recurrent layer whose 7×7 filters are constrained to behave like spatial derivatives, so that branch carries smooth physical motion. A 3-layer ConvLSTM, with hidden sizes 128, 128 and 64, models whatever that physics cell cannot explain. The two branch outputs are added and decoded to one frame. That frame is fed back in, and the loop runs once per future frame.

It has 3.1M parameters, close to our 3.34M, and 15.3G FLOPs, because both branches run at every step. Its published Moving MNIST MSE is 28.19. It is the more accurate model at this parameter count, and it costs about five times the compute.

### The two other STLight rows

Trained XS and paper STLight-S are not other designs.

- Trained XS is the same five-stage network with `dim` 224. It has 1.05M parameters, 0.87G FLOPs, and MSE 40.00. S beats it on MSE, MAE, and SSIM.
- Paper STLight-S is the same network with `dim` 1000: 17.1M parameters, 16.5G FLOPs, MSE 23.29. That is the width in Table 3. The run we have is `dim` 424.

## Why the parameter count is lower

Present this as an accident in reading the table. The architecture was not changed.

Table 3 prints STLight-XS as 11.1M parameters and STLight-S as 17.1M. Those were first read as 1.1M and about 3.3M. `dim` was then solved to hit those wrong targets: 224 gives 1.05M, and 424 gives 3.34M. The paper widths are `dim` 800 and `dim` 1000. The block, the data, and the 200-epoch recipe are the paper's. Scaling the measured FLOPs by those widths lands on the paper's 10.6G and 16.5G, so the implementation matches and only the channel count was short.

The sentence to say: Table 3 lists STLight-S at 17.1 million parameters. We trained 3.34 million because 11.1 million was first read as 1.1 million. A full-width rerun is about four days on the same GPU, which does not fit the deadline.

What the finished runs still show is a 200-epoch model. At 3.34M parameters it sits in the same size band as SimVP+ConvMixer and PhyDNet. The step from trained XS to trained S shows that making this network wider improves MSE, MAE, and SSIM.
