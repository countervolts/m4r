# MetalFX ML temporal scaler ("BBR") — what is known

Source: static reading of macOS 27.2.1 (26B131) MetalFX: the arm64e binary (`BBRNet_Filter<MFXDevice3>`,
`tBBRNet`, `sBBRNet`, `Conv*` classes) and the kernels' AIR metadata. Nothing here was observed running on a
Mac. **Verified** means checked against code/metadata in two independent places or on our GPU; **inferred**
means read from one place or from names. Numeric constants (activation scales, weight offsets) are not written
here; `tools/` extracts them from the user's own binary.

## Effect and filter

`_MFXTemporalScalingEffectBBR` owns a C++ `BBRNet_Filter<MFXDevice3>`. `encode()` runs:

1. `encodePre` (debug group `MetalFX_Temporal_BBR_PreProcessing`)
   - auto exposure: `luma_log_sum` (grid {32,64}x tg {8,4}, threadgroups) → `luma_log_sum_to_exposure`
     ({1} x {1024}); otherwise `exposure_texture_to_buffer` with the game's exposure texture.
     Exposure lives in two 2-byte buffers used ping-pong (index flips each frame), initialised to half 1.0.
   - `bbrnet_build_tensors_low_resolution`: textures 0 color, 1 depth, 2 motion, 3 → +0x1e8 (RGBA16F, input
     size), 4 → +0x1e0 (RG16F, input size); bytes(0) = ±1.0 (reversed depth); params (transient buffer 1);
     bytes(5) jitter delta when jittered motion vectors are enabled; bytes(2) preExposure; buffers 3/4 exposure
     ping-pong; grid (input + 2) / 2 in 8x8 (inferred from the arithmetic).
   - `buildSpatialTensor`: buffer 0 → +0xc0 (sBBRNet input), texture 0 → +0x1e8.
   - `buildTemporalTensor`: textures 0 +0x1e8, 1 +0x1e0, 2 +0x248 (history), 4/6 ping-pong +0x218/+0x220,
     8 +0x228, 9 +0x230; buffers 0 +0xd0, 2 +0xe8, 3/4 exposure, 5 +0x278, 6 +0xd8, 7 +0xe0 (tBBRNet inputs);
     grid ((size+15)/2 & ~7) in 16x16.
2. networks (fused path, flag at +0x1ca): tBBRNet then sBBRNet (below). Otherwise MPSGraph (not ported).
3. `encodeMid`: `brnetv3_flow_splat_init_forward_z` (unless reversed depth), `BBRNet_flow_splat`,
   `brnetv3_flow_splat_texture`, `BBRNet_flow_diff_texture`, `BBRNetWarpHistory` (grid output w x (h+3)/4,
   tg {16,8}), `BBRNetComputeOvershootClamp` (output size, tg 16x16).
4. `encodePost`: `BBRNetAnisoGaussian` writes the output; `brnetv3_copy_reactive_mask` when a reactive mask is
   bound.
5. An asynchronous "overlap" network (`sacneh.mtlpackage`, MPSGraph) on the queue; can be disabled with
   `MTLFX_DISABLE_OVERLAP_NETWORK`. Role not determined.

`BRNetShaderParams` (144 bytes, field names from kernel metadata): contentSize, maxInputSize, outputSize,
color/motion/depth/reactive/output offsets (ushort2 each), jitterOffset, motionScale, outputResMotionMultiplier
(= content / output size), normalizeMotionScale (= motion scale / input size; Apple uses a two-step
`frecpe`/`frecps` reciprocal), and five uint4 tensor-stride vectors.

## Intermediate resources (allocated in `init`)

| offset | Metal format | size |
|---|---|---|
| +0x1e0 | RG16Float | input |
| +0x1e8 | RGBA16Float | input |
| +0x1f0, +0x1f8, +0x248 | RGBA16Float | output |
| +0x218, +0x220 | R16Float | output / 2 (ping-pong) |
| +0x228 | R8Unorm | output |
| +0x230 | RG8Unorm | output / 2 |
| +0x238, +0x240 | RG8Unorm | output |
| +0x208, +0x210 | R8Unorm | output (line locking only) |
| +0x200 | reactive format | output (reactive mask only) |
| +0x258 / +0x260 | RG16Float / R8Unorm | input |
| +0x250 | buffer, input w·h·4 bytes | flow splat |
| +0x278 / +0x280 | buffers, 1 / 8192 bytes | |

## tBBRNet (temporal network)

Weights: `emit_tbbrnet_nhwc_q_constants.dat`, which must be exactly 288,576 bytes (checked by the loader). Eleven
int8 3x3 convolutions forming a U-Net with additive skips (verified: the per-layer offsets account for every
byte of the file):

| layer | kind | in → out ch | resolution |
|---|---|---|---|
| 1 | Conv3x3 stride 1 (u8 + f16 input feeds) | 32 → 32 | full (network input) |
| 2 | Conv3x3 stride 2 | 32 → 32 | 1/2 |
| 3 | Conv3x3 stride 2 | 32 → 64 | 1/4 |
| 4 | Conv3x3 stride 2 | 64 → 128 | 1/8 |
| 5 | Conv3x3 stride 1 | 128 → 64 | 1/8, output to intermediate storage |
| 6 | Conv3x3 stride 1, bilinear ↑2 + skip(layer 3) | 64 → 64 | 1/4 |
| 7 | Conv3x3 stride 1 | 64 → 32 | 1/4 → storage |
| 8 | Conv3x3 stride 1, ↑2 + skip(layer 2) | 32 → 32 | 1/2 |
| 9 | Conv3x3 stride 1 | 32 → 32 | 1/2 → storage |
| 10 | Conv3x3 stride 1, ↑2 + skip(layer 1) | 32 → 32 | full |
| 11 | Conv3x3 stride 1, head (writes target, temporalTex, rnnTex) | 32 → 32 (second half of a 64-output conv) | full |

Per layer the file holds `u8 W[kh·kw·C][K]` (output channel innermost), `fp16 scale[K]`, `u8 zeroPoint[K]`,
`fp16 bias[K]`, packed back to back. The host does not reorder weights; it slices them and folds the previous
layer's activation zero point into the bias (`mfx::weights::adjustBiasForInputActZp`):

    bias'[k] = half( fma(actScale · scale[k] · (−actZp), Σ_j W[j][k] − zp[k] · kh·kw·C, bias[k]) )

computed in f32 (the sum exactly as u32). Activations are u8 with a per-tensor scale and zero point (the
layers use LeakyReLU, output zero point 3.0); the first layer's input is quantised with scale 1/255 and the last
layer re-quantises to 1/255. Layer 1 additionally gets its weights dequantised to fp16 and split into two
16-channel halves (`dequantizeUint8WeightsToFP16`, `_bufWeightsFirst16/Last16`) for the u8+f16 feed path.

Kernels and their function constants (names from `!air.function_constants`):
`Conv3x3Stride1U8` / `Conv3x3Stride1Enc0FQSplit`: H, W, K, C, QsI, QsRinv, useBilinear, DqsBilinearImage, useSkip,
DqsSkip, CSkip, NmatsHW, NmatsK, outputBlockK, inputIsQuantized, outputToTexture, useSigmoid, useD2S,
useLeakyReLU, outZp, combinedInZp, bilinearImageZp, skipZpAct, blockMajorWeights, headTemporalTex, headRnnTex,
inputIsSigned. The layer object copies a `Conv3x3Stride1Descriptor`; constant ↔ descriptor offsets are in
`tools/bbr_params.py`.

The network runs at **output size / 2** (`getDefaultBBRNetDescriptor(version, outputWidth, outputHeight)`, then
H = h/2, W = w/2); layer 11 ends with sigmoid + depth-to-space.

### Layer execution (verified from the layer classes)

- Stride 1 (`Conv3x3Stride1U8`): buffers 0 Image, 1 Weights, 2 Result, 3 Weights_scale, 4 Biases, 5 ZeroPoints,
  6 Skip. A SIMD group computes `NmatsHW·16` pixels × `NmatsK·16` output channels; threadgroup = 32 · desc[0x6a]
  threads; threads = roundup(ceil(H·W/(16·NmatsHW)) · ceil(K/(16·NmatsK)) · 32, tg). Weights uploaded as
  `toBlockMajor<uchar>(W, C, K, 3, 3)`: per tap, the [C][K] matrix cut into 16x16 tiles ordered (c/16, k/16),
  each tile row-major. Activations are plain NHWC u8 (`outputBlockK` = `inputBlockK` = 0).
- Skip layers (6, 8, 10) run two dispatches: `BilinearUpsample2xSkipU8` (feed, skip → private scratch of
  H·W·C bytes; grid H·W rounded to 256, tg 256), then the conv with useBilinear = useSkip = false.
- Stride 2 (`Conv3x3Stride2U8`, MXU version 1 only): buffers 0–5 as above; weights uploaded unchanged (HWIO
  u8); threads = roundup(roundup(K,32) · ceil(Hout·Wout/32), 128), tg 128.
- Layer 1 on Apple's default path (filter byte +0x11 = `q_temporal_in` = 1) is `Conv3x3Stride1Enc0FQSplit`:
  input channels 0–15 come from an fp16 tensor with fp16 weights (`dequantizeUint8WeightsToFP16`:
  half(f32(W − zp) · scale)), channels 16–31 from a u8 tensor with u8 weights; both `toBlockMajor` with C = 16;
  bias unfolded.
- Layer 11 (head): `executeWithCommandBuffer:feed:target:temporalTex:rnnTex:`, output → the network output
  buffer, which `buildTemporalTensor` reads back next frame (recurrent) and `BBRNetAnisoGaussian` consumes.

### sBBRNet

Five per-pixel layers of 64 inputs (K = 64, 64, 64, 64, 128), weights [64][K], tables `kLayerK`,
`kWeightOffsets`, `kBiasOffsets`, `kScaleOffsets`, `kZPOffsets`, `kActZP`, `kActScale` in the binary; bias folded
as for tBBRNet with the previous layer's activation zero point and scale (layer 0: zero point 0 and a fixed
input scale). Kernel `SBBRNet`.

## End-to-end result

`tests/run_bbr.cpp` runs the whole plan (`tools/prepare_bbr.py`: exposure, tensor builds, tBBRNet, sBBRNet, flow
splat/diff, history warp, overshoot clamp, AnisoGaussian composite, sacneh prefilter; 27 passes) on the
moving-camera fixture from `tests/scene_motion.py` (adapted from x4r): 640x360 → 1280x720, 8 frames,
**28.27 dB** sRGB PSNR against the 2x2-supersampled reference on the last frame (bilinear: 21.9 dB), 1.7 ms per
frame on an RX 7700 XT (5.3 ms with the first, dot-product GEMM lowering; see docs/mxu.md).

- `jitterOffset` is the **negative** of the sub-pixel offset the frame was rendered with (with the positive sign the
  same run gives 22.7 dB and a visibly blurred, offset history).
- `motionVectorScale` (1, 1) with vectors in input pixels pointing to the previous position, as the header says.
- The `sacneh` overlap network is a separable 19-tap filter (taps symmetric, alternating, summing to 1: an inverse
  B-spline prefilter) on the 2-plane fp16 history tensor, from AnisoGaussian's output (+0xf8) to WarpHistory's input
  (+0xf0); m4r runs it as its own GLSL pass (`shaders/sacneh.comp`) with the taps read from
  `emit_sacneh_constants.dat`.

## Which scaler a Mac gets, GPU or Neural Engine

`-[MTLFXTemporalScalerDescriptor newTemporalScalerWithDevice:]` (read from the arm64e disassembly) chooses the
effect from the descriptor's `version` (after the `MTLFX_TEMPORAL_SCALER_VERSION` override, which is only read in
internal builds):

| version | effect | execution labels in the binary |
|---|---|---|
| 0x10000 | `_MFXTemporalScalingEffectV3` | `TAAUv3 \| ANE`, `TAAUv3 \| GPU` |
| 0x10001 | `_MFXTemporalScalingEffectNRS` | |
| 0x10002, 0x10003 | `_MFXTemporalScalingEffectV4` (`BRNet4_1`, packages `brnet`/`brnet_ane`, `_useANE` flag) | `TAAUv4 \| ANE`, `TAAUv4 \| GPU` |
| 0x10004, 0x10005 | `_MFXTemporalScalingEffectBBR` | `TAAUBBR \| GPU` only |
| anything else (default) | BBR if `deviceSupportsFusedBBRNet(device, pixels)`, otherwise V4 | |

`deviceSupportsFusedBBRNet` requires `supportsFamily:1010` (0x3f2; read as the newest Apple GPU family, the one with
the matrix instructions m4r lowers), then a device property (read as the GPU core count) above 20, or equal to 20
with at most 4,096,000 pixels. So the newest GPUs get BBR on the GPU's matrix units, and other Macs get V4, which
has the Neural Engine mode (`MTLFX_FORCE_GPU` and `MTLFX_EXECUTION_MODE` override it). Apple's own order is
V3 < V4 < BBR, so BBR is the latest of the three.

m4r runs BBR: Apple's network (`tBBRNet`, `sBBRNet` hand-written Metal kernels in `default.metallib`) and weights,
on the GPU, nothing substituted. It does not run V3, V4 or the denoiser, which are the effects with Neural Engine
variants. Not verified: the property read by `deviceSupportsFusedBBRNet` is the core count; whether V4 defaults to
the ANE or the GPU (`_useANE` is not decoded); whether V4's ANE and GPU paths agree numerically. Nothing here was
observed running on a Mac.

The frame interpolator has no network at all (docs/metalfx-framegen.md): optical flow and warps on the GPU.

## Open

- encodePost details, the `sacneh` overlap network
- buffer sizes chosen by `tBBRNet setupBuffers` (allocated generously for now)
