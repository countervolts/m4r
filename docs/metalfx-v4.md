# MetalFX temporal scaler V4 — what is known

Static reading of macOS 27.2.1 MetalFX (same sources and conventions as docs/metalfx-bbr.md). **Nothing of V4 runs in m4r yet**; this
page is the state of the reverse engineering, with inferred parts marked.

## Where V4 sits

`_MFXTemporalScalingEffectV4` (and the Metal 4 twin `_M4FXTemporalScalingEffectV4`) is chosen by
`-[MTLFXTemporalScalerDescriptor newTemporalScalerWithDevice:]` for descriptor versions 0x10002 and 0x10003, and as the fallback for
the default version when `deviceSupportsFusedBBRNet` says no (docs/metalfx-bbr.md, "Which scaler a Mac gets"). It is the effect with
the `TAAUv4 | ANE` and `TAAUv4 | GPU` execution labels. Its host stages are named `MetalFX_Temporal_PreProcessing`,
`MetalFX_Temporal_MidProcessing`, (Metal 4 only: `MetalFX_Temporal_FusedNetwork_Wait` / `_Signal`) and
`MetalFX_Temporal_PostProcessing`; the network is the class `BRNet4_1`. Its weights are `emit_v41_nhwc_constants.dat` /
`emit_v41_nchw_constants.dat` in the framework's Resources (`BRNet4_1 setupModelWithLibrary:` loads them), and the MPSGraph packages
`brnet` / `brnet_ane` (`model_0`, `model_1`) are the alternative executions (GPU graph / Neural Engine).

## The network (`BRNet4_1`)

Ivars: `conv12x12_`, `conv3x3Stride2_1_ .. _3_`, `conv1x1Fused_1_ .. _4_`, `conv3x3Sigmoid_`, buffers `bufSkip0_ .. 2_`,
`bufInterm0_ / 1_`. `-executeWithCommandBuffer:feed:target:intermediates:` runs (feed → skip → target):

| # | layer | feed | skip | target |
|---|---|---|---|---|
| 1 | `Conv12x12Stride4` | input tensor | | bufSkip2 |
| 2 | `Conv3x3Stride2` 1 | bufSkip2 | | bufSkip1 |
| 3 | `Conv3x3Stride2` 2 | bufSkip1 | | bufSkip0 |
| 4 | `Conv3x3Stride2` 3 | bufSkip0 | | bufInterm0 |
| 5 | `Conv1x1FusedBilinear` 1 | bufInterm0 | none | bufInterm1 |
| 6 | `Conv1x1FusedBilinear` 2 | bufInterm1 | bufSkip0 | bufInterm0 |
| 7 | `Conv1x1FusedBilinear` 3 | bufInterm0 | bufSkip1 | bufInterm1 |
| 8 | `Conv1x1FusedBilinear` 4 | bufInterm1 | bufSkip2 | bufInterm0 |
| 9 | `Conv3x3SigmoidD2S` | bufInterm0 | none | output tensor |

(a U-Net shaped encoder / decoder: the three stride-2 outputs are the skips, each fused 1x1 layer upsamples bilinearly and adds
its skip; the last layer is a 3x3 with a sigmoid and a depth-to-space.) Between layers the Metal 4 path puts a barrier
on the buffer just written.

### Layer descriptors (`GenerateDefault*Descriptor(device, weights NSData, H, W)`)

Read from the arm64 code; field meaning follows the layer classes' `setupMetalWithLibrary:` (not read yet for V4, so the names are
guesses except where stated). Offsets are into the weight file `emit_v41_*.dat`.

| layer | channels (in → out) | size | weights | other slices | float pair at +0x18 / +0x20 |
|---|---|---|---|---|---|
| 12x12 stride 4 | 5 → 32 (`0x00050020`) | H/4, W/4 | 0xbe0, 0x5a00 B (= 144 taps x 5 x 32 u8) | 0x590 (64 B), 0x500 (64 B) | 0.007843 (= 2/255), 8.406 |
| 3x3 stride 2 (1) | 32 → 32 (`0x00200020`) | H/8, W/8 | 0x65e0, 0x2400 B (= 9 x 32 x 32) | 0x5d0 (64 B), 0x4c0 (64 B) | 0.1190, 7.578 |
| 3x3 stride 2 (2) | 32 → 64 (`0x00200040`) | H/16, W/16 | 0x89e0, 0x4800 B (= 9 x 32 x 64) | 0x610 (128 B), 0x440 (128 B) | 0.1320, 5.877 |
| 3x3 stride 2 (3) | 64 → 128 (`0x00400080`) | H/32, W/32 | 0xd1e0, 0x12000 B (= 9 x 64 x 128) | 0x690 (256 B), 0x340 (256 B) | 0.1702, 3.018 |
| 1x1 fused (1) | | H/32, W/32 | 0x1f1e0 (16 KB), 0x231e0 (8 KB) | 0x810 (256 B), 0x910 (128 B), 0x240 (256 B), 0x1c0 (128 B) | pairs 0.3313 / 0.1927 and 5.188 / 4.171 |
| 1x1 fused (2..4), 3x3 sigmoid | see `tools/` (to be done) | | | | |

The two 64 B tables of the stem and the paired slices of the other layers look like per-channel scales and biases (fp16) and
zero points; the float pairs look like activation scales / zero points. All inferred.

## Kernels

Layer kernels: `Conv12x12Stride4(MXU3_K32)`, `Conv3x3Stride2(MXU3_K32 / K64, U8)`, `Conv1x1FusedBilinear(MXU3_K16 / K32 / K64)`,
`Conv3x3SigmoidD2S(MXU3_K32 / K64)`, `DepthToSpaceWeights`; pre / post kernels `brnetv3_*` / `brnetv34_*`
(`build_tensors_low_resolution`, `lowres_signals`, `update_sample_statistics`, `color_bounding_box_variance`,
`warp_history_high_resolution`, `anisotropic_gaussian_filter`, `output_spatial_lr`, `output_temporal`, ...) and tile kernels
`unified_pass_tile` / `unified_low_resolution_tile`.

All of these translate for gfx1101 with default settings **except the two tile kernels**: the metadata parser fails on their
argument structs (`struct_info` in `air/m4r_air.py`, `IndexError` on a nested member list).

## What is missing

1. Fix the metadata parser for the tile kernels and check that the imageblock emulation covers what they use.
2. The layer classes' setup (function constants, buffer layouts, weight layout conversions as `block_major` for BBR) and the
   remaining four descriptors.
3. The effect's `encodeToCommandBuffer:`: the pre / mid / post pass lists with their bindings and parameter blocks
   (about 2,350 instructions).
4. A plan generator (`tools/prepare_v4.py`), a way to select the model in the m4r API and in OptiScaler's menu, and a check of
   the result against a reference (PSNR on the motion fixture, as for BBR).

Whether V4 would look better than BBR is open: its version number is lower than BBR's, Apple's default picks BBR where it is
available, and V4's ANE variant is only an execution target, not a quality difference (and m4r would run its GPU path anyway).
