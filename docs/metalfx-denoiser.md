# MetalFX temporal denoised scaler — what is known

Source: static reading of macOS 27.2.1 (26B131) MetalFX (`_MFXTemporalDenoisingScalingEffect`, `DBFNet`, `UBFNet`, the
`default.metallib` kernel list, the two `emit_*net_v3_2_constants.dat` files and the `.mtlpackage` graphs). Nothing was
observed running. **Verified** = checked in two independent places; **inferred** = from names or sizes. m4r does not run
any of this yet.

## It is a denoiser *and* an upscaler, not BBR

The effect's debug groups give the GPU-path order: `MetalFX_Denoiser_DPreProcessing` → `DMidProcessing` → `DNet` →
`DPostProcessing_UPreProcessing` → `UMidProcessing` (`MetalFX_Upscaler_UMidProcessing`) → `UNet` →
`MetalFX_Upscaler_UPostProcessing`. D = denoise net (DBFNet), U = upscale net (UBFNet: its head is a 3x3 conv to 44 channels,
depth-to-space, sigmoid = 11 channels at 2x). No BBR (tBBRNet/sBBRNet) is involved. `initWithDevice:` builds a
`BRNet_v3_Filter<MFXDevice3>` (V3 temporal machinery; weights `emit_v35_constants.dat`, 156,832 B; kernels `brnetv3_*`) and a
`BFNet_v1_Filter` that wraps DBFNet and UBFNet. The log strings `[MetalFX] dBFNet/UBFNet fused-MXU path ACTIVE (Metal3;
NHWC-native)` say the GPU path runs Apple's hand-written fused classes (same style as BBR); if either init fails the scaler
creation fails (there is no MPSGraph fallback after the layout change). The MPSGraph packages (`dbf`, `ubf`, `*_ane`,
`brnet*`) are the Neural Engine path ("Denoising | ANE" vs "Denoising | GPU" in the HUD; `MTLFX_FORCE_GPU`,
`MTLFX_EXECUTION_MODE`; `denoise_brnet_ip/op` are tensor-dump names on that path).

## Parameters (fp16)

| net | parameters | file |
|---|---|---|
| DBFNet | ~199k (13 in, 7 out channels) | `emit_dbfnet_v3_2_constants.dat`, 399,374 B |
| UBFNet | ~152k (5 in, 11 out at 2x) | `emit_ubfnet_v3_2_constants.dat`, 305,272 B |
| BRNet v3.5 | ~78k (inferred: 156,832 B / 2) | `emit_v35_constants.dat` |

## Layer tables (verified)

The `.dat` files are the same tensors as the packages' `resources.bin`, fp16, HWIO, each conv stored as `[bias][weights]` in
network order. The offsets below are the ones the cell constructors in `DBFNet/UBFNet setupModelWithLibrary:` pass (read from
the disassembly) and the ones `tools/mpsgraph.py --dat` finds by byte search; they agree. Every conv is followed by
bias + ReLU except the last of each net.

DBFNet — cells E1..E4 are `DBF_GenerateEncoderCell` (3x3 conv, 1x1 conv; class `MaxPoolConv`, 2x2 max pool after E1-E3),
D1..D3 are `DBF_GenerateDecoderCell`. Upsampling is x2 and added to the skip (not concatenated).

| cell | convs | bias @ | weights @ |
|---|---|---|---|
| E1 | 3x3 13→32, 1x1 32→32 | 0x0, 0x1d80 | 0x40, 0x1dc0 |
| E2 | 3x3 32→64, 1x1 64→64 | 0x25c0, 0xb640 | 0x2640, 0xb6c0 |
| E3 | 3x3 64→64, 1x1 64→64 | 0xd6c0, 0x1f740 | 0xd740, 0x1f7c0 |
| E4 | 3x3 64→64, 1x1 64→64 | 0x217c0, 0x33840 | 0x21840, 0x338c0 |
| D1 | 3x3 64→64, 1x1 64→64 | 0x358d0, 0x47950 | 0x35950, 0x479d0 |
| D2 | 3x3 64→64, 1x1 64→32 | 0x499e0, 0x5ba60 | 0x49a60, 0x5baa0 |
| D3 | 3x3 32→32, 1x1 32→16, 1x1 16→7 | 0x5cab0, 0x612f0, 0x61710 | 0x5caf0, 0x61310, 0x6171e |

UBFNet — E0, E1 are 6x6 **stride 2** (`ConvKxKStride2`, pad 2), E2, E3 3x3 stride 1 (same class), E4 `MaxPoolConv`
(64→128 3x3), then `UBF_GenerateHeadOnly` cells (1x1 convs, bilinear x2 + skip), then `UBF_GenerateHead`
(3x3 16→44, depth-to-space, sigmoid; class `Conv3x3Stride1`).

| cell | convs | bias @ | weights @ |
|---|---|---|---|
| E0 | 6x6 s2 5→8 | 0x0 | 0x10 |
| E1 | 6x6 s2 8→32 | 0xb50 | 0xb90 |
| E2 | 3x3 32→32 | 0x5390 | 0x53d0 |
| E3 | 3x3 32→64 | 0x9bd0 | 0x9c50 |
| E4 | 3x3 64→128 | 0x12c50 | 0x12d50 |
| bottleneck | 1x1 128→128, 1x1 128→64 | 0x36d50, 0x3ee50 | 0x36e50, 0x3eed0 |
| U1 | 1x1 64→64, 1x1 64→32 | 0x42ee0, 0x44f60 | 0x42f60, 0x44fa0 |
| U2 | 1x1 32→32, 1x1 32→16 | 0x45fb0, 0x467f0 | 0x45ff0, 0x46810 |
| U3a/b | 1x1 16→32, 1x1 32→16 | 0x46c20, 0x47060 | 0x46c60, 0x47080 |
| U3c | 1x1 16→16 | 0x47480 | 0x474a0 |
| head | 3x3 16→44 | not found as plain bytes | 0x476f8 (6,336 B); the 6,336 B after it hold the head's bias/permuted copy (not decoded) |

The skips (UBF): 64 ch after E3, 32 ch after E2, and a 16-channel strided slice of the 32-channel level, sliced to
[1,16,136,240] in the graph (why a fixed spatial size is baked in is unexplained).

## Resolved questions

- **Resize mode**: bilinear (inferred: the fused kernels are `BilinearUpsample2xSkipF16` and `Conv1x1FusedBilinear`; the kernel IR
  has not been read for the alignment convention).
- **UBF strides**: E0/E1 stride 2, otherwise 1 (verified in `UBF_GenerateEncoder` arguments). The `mps` conv properties in the
  package graph carry no stride; `tools/mpsgraph.py` does not decode them and the package graph is not the source of truth.
- **brnet packages**: `brnet`/`brnet_ane` are quantized MPSGraph models (V4/ANE side); on the GPU path the denoiser uses
  DBFNet + UBFNet + the V3 filter only.

## Inputs (public header `MTLFXTemporalDenoisedScaler.h`)

color, depth, motion, diffuseAlbedo, specularAlbedo, normal, roughness; optional specularHitDistance, denoiseStrengthMask,
transparencyOverlay, reactiveMask; exposureTexture or auto exposure, preExposure, jitterOffsetX/Y, motionVectorScaleX/Y,
shouldResetHistory, depthReversed, worldToViewMatrix, viewToClipMatrix. This is the DLSS Ray Reconstruction input set, so
OptiScaler's DLSS-D (`upscalers/dlssd`) path already receives all of it. Cyberpunk on Linux is the Windows build: it has no
Metal path, so nothing needs to look like an M-series chip. The game offers Ray Reconstruction when OptiScaler presents
DLSS-D; the MetalFX denoiser would be a new backend behind that (a patch 0003 next to 0001/0002).

## Host logic: what the algorithm is (2026-11-19, from kernel signatures and the filter classes)

Kernel argument names below are the AIR `air.arg_name` metadata (certain); how they are wired in the encode functions is
only partly traced (see the end of this section). `tools/host_trace.py` prints the per-dispatch bindings of any encode
function (pipeline ivar, `setTexture` slot <- source, buffers, `setBytes` lengths).

**The denoiser is learned guidance + SVGF-style filtering, not a network that outputs colour.** DBFNet's three outputs
(`dbfnetv1_output_channels`) are `outputLearnedFeatureArray`, `outputNoiseProbBandwidth` and
`outputNetworkDenoiserTemporalWeight`. They steer a temporal reprojection and an edge-aware filter:

| stage (debug group) | work | kernels (`BFNet_v1_Filter` method) |
|---|---|---|
| DPreProcessing | exposure for the denoiser; build the 13-channel DBF input and the internal depth/motion, mirror mask | `luma_log_sum_denoiser`, `luma_log_sum_to_exposure_denoiser` (`BRNet_v3_Filter::encodeExposureCalcForDenoiser`), `dilateFlow`, V3 flow kernels (`encodePreForDenoise`), `dbfnetv1_lowres_signals` (`BFNet::encodePre`, pipeline ivar +0x38) |
| DNet | DBFNet (cells E1-E4, D1-D3) | `MaxPoolConv`, `BilinearUpsample2xSkipF16` and friends |
| DMidProcessing | split network outputs, reproject history | `dbfnetv1_output_channels` (+0x40), `reproject_warp` (+0x48, `BFNet::encodeMid`) |
| (atrous) | edge-aware à-trous filter iterations and temporal blend | `bilateral` (+0x60, `encodeAtrous`: `step_distance`, `iteration`, `history_invalid`), `reproject_blend` (+0x50), `update_vbbr` (+0x70) |
| DPostProcessing_UPreProcessing | remodulate with albedo, merge with the noisy colour by `in_str_mask`, tensors for the upscaler | `denoiser_between_processing` (+0x68) |
| UMid/UNet/UPost (Upscaler) | UBFNet (5 in, 11 out at 2x) in a V3-style temporal upscaler | `brnetv3_lowres_signals`, `brnetv3_output_temporal`/`output_spatial_lr` (read the network output), `brnetv3_warp_history_high_resolution`, `brnetv3_anisotropic_gaussian_filter` (output) |

Inputs the kernels read (AIR names): `texNormal`, `texMotion`, `texDepth`, `texSpecularHitDistance`, `texSpecularAlbedo`,
`texDiffuseAlbedo`, `texRoughness`, `texInputFrame`/`texPrevInputFrame`, a ping-pong `mirrorMask`, `exposure`/`preExposure`,
and in `denoiser_between_processing` also `in_noisy_color`, `in_str_mask` (the denoise strength mask), `in_compose_buffer`,
`in_shininess`. `reproject_warp` keeps per-pixel state across frames: frame count, running mean, boost-history mask,
demodulation Y, previous normal/roughness, previous colour, shininess. `svgf_prepare_depth_normal` (SVGF is in the name)
prepares depth+normal for the filter.

Formats read from the `BFNet_v1_Filter` constructor (MTLPixelFormat: 0x73 RGBA16F, 0x7d RGBA32F, 0x37 R32F, 0x19 R16F,
0x41 RG16F, 0x0a R8Unorm): intermediate textures at ivars +0x78..+0xc8 and +0xd0..+0x158 (history/ping-pong); the exact
size and format per ivar is not tabulated yet.

## The GPU path of `encodeToCommandBuffer:`, in order (decompiled with Ghidra, 2026-11-19)

The effect owns two C++ filters: `_filter` (`BRNet_v3_Filter`, the V3 temporal machinery) and `_denoiseFilter`
(`BFNet_v1_Filter`, the SVGF-style denoiser), plus two fused nets (`dbfnetFused`, `ubfnetFused`; MPSGraph wrappers
`dbfnet_net_wrapper`/`net_wrapper` on the other path) and buffers `dbfnet_input_TensorData`, `dbfnet_output_TensorData`,
`input_TensorData`, `output_TensorData`. Textures it owns: `denoiserColorTexture` (the denoised colour at render size),
`preUpscaleComposeTexture`, `history`, `prevReactiveTexture`. Per frame it flips the encode index of both filters (ping-pong
of exposure buffers and mirror masks), computes `prevViewProjectionMatrix` from the current `viewToClip`·`worldToView`, and
builds the shared params struct (input/content/output size, input/output ratio, normalised motion scale, jitter).
The call islands below are identified by signature and argument order, not by symbol (their targets are not in the file).

| # | encoder (debug group) | call | what it is |
|---|---|---|---|
| 1 | DPreProcessing | `BRNet_v3_Filter::encodeExposureCalcForDenoiser(color, exposureTexture, reset, autoExposure, params)` | `luma_log_sum_denoiser`, `luma_log_sum_to_exposure_denoiser` into the filter's exposure buffers |
| 2 | DPreProcessing | `BFNet_v1_Filter::encodePre(normal, motion, color, specularAlbedo, diffuseAlbedo, roughness, depth, specularHit, buffers: dbfnet_input_TensorData + exposure, preExposure, reset, prevViewProj, worldToView, params)` | `dbfnetv1_lowres_signals` (13 textures, 3 byte buffers): fills the DBF input tensor, internal depth (+0xc8), internal motion (+0xc0), mirror mask (ping-pong, +0xd8) |
| 3 | DMidProcessing | `BRNet_v3_Filter::encodeMidForDenoise(internalDepth, internalMotion, specularHit, mirrorMasks, reversedDepth, params)` | V3 flow side (dilate/splat): produces the dilated flow (+0xd0) |
| 4 | DMidProcessing | `BFNet_v1_Filter::encodeMid(roughness, diffuseAlbedo, specularAlbedo, internalMotion, color, internalDepth, specularHit, normal, flow, ...)` | `reproject_warp` (pipeline +0x48) |
| 5 | DNet | `DBFNet executeWithCommandBuffer:` from `dbfnet_input_TensorData` to `dbfnet_output_TensorData` | the fused DBFNet (E1-E4, D1-D3) |
| 6 | DPostProcessing_UPreProcessing | `BFNet_v1_Filter::encodePost(dbfnet_output_TensorData, reset)` | `dbfnetv1_output_channels` (learned features, noise prob/bandwidth, temporal weight) |
| 7 | (same) | `BFNet_v1_Filter::encodeAtrous(color, internalMotion, internalDepth, normal, roughness, diffuseAlbedo, specularAlbedo, denoiseStrengthMask, preUpscaleCompose, -> denoiserColorTexture, flow, ...)` | `bilateral` iterations, `reproject_blend`, `denoiser_between_processing` |
| 8 | (same) | `BRNet_v3_Filter::encodePreForDenoise(denoiserColor, internalDepth, internalMotion, history, exposure, normal, diffuseAlbedo, roughness, specularAlbedo, buffer: input_TensorData, flags, params)` | `brnetv3_lowres_signals`/build tensors: UBF input |
| 9 | UMidProcessing | `BRNet_v3_Filter::encodeMidForDenoise(denoiserColor, history, internalMotion, reversedDepth, params)` | V3 flow/history prep for the upscale |
| 10 | UNet | `UBFNet executeWithCommandBuffer:` from `input_TensorData` to `output_TensorData` | the fused UBFNet |
| 11 | UPostProcessing | `BRNet_v3_Filter::encodePost(output_TensorData, outputTexture, reactiveMask, prevReactiveTexture, history, reset, params)` | `brnetv3_output_temporal/spatial_lr`, `brnetv3_warp_history_high_resolution`, `brnetv3_anisotropic_gaussian_filter` -> `outputTexture` |

Where the decompiled source lives: Ghidra headless import of `bins-27/MetalFX` plus a decompile script (project and
output under the session scratchpad, not in the repository); the Device4 (Metal 4) copies of each method are
identical in structure.

## Parameter structs and function constants (2026-11-19)

Layouts are from the AIR struct metadata (field names and offsets are certain); where each field comes from is read from the
decompiled host code.

`BRNetShaderParams` (144 B, bound by `setBytes` as `params`/`shaderParams`; built once per frame by the effect):

| offset | field | value |
|---|---|---|
| 0 | `contentSize` ushort2 | (inputWidth, inputHeight) |
| 4 | `maxInputSize` ushort2 | same |
| 8 | `outputSize` ushort2 | (outputWidth, outputHeight) |
| 12..31 | `colorOffset`, `motionOffset`, `depthOffset`, `reactiveOffset`, `outputOffset` ushort2 | 0 (no sub-rectangles) |
| 32 | `jitterOffset` float2 | the effect's `jitterOffset` |
| 40 | `motionScale` float2 | the effect's `motionVectorScale` |
| 48 | `outputResMotionMultiplier` float2 | (inputW / outputW, inputH / outputH) |
| 56 | `normalizeMotionScale` float2 | motionScale / (inputW, inputH), reciprocal as `frecpe` + `frecps` steps (as in BBR) |
| 64..143 | five `uint4` tensor strides | not set by the effect; filled elsewhere (not found yet) |

`dbfnet_v1_build_tensors_t` (40 B, `dbfnetv1_lowres_signals` buffer 0, `dbfnetv1_output_channels` buffer 0), from `BFNet::encodePre`:
`motionVectorScale` float2 @0 = params.motionScale, `prevJitter` float2 @8 = the previous frame's `jitterOffset` (stored in the
filter at `this+0x30` at the end of each frame), `jitter` float2 @16 = params.jitterOffset, `rowBytes` int @24 = the filter's
tensor width (`this+0x18`), `height` ushort @28, `width` ushort @30 (`this+0x24`, `this+0x20`), `isHistoryInvalid` bool @32 =
the call's reset flag.

`brnet_v3_reflection_depth_params_desc_t` (224 B, `dbfnetv1_lowres_signals` buffer 5), also from `encodePre`:
`invViewProjectionMatrix` @0 = inverse of the current view-projection (a 4x4 inverse helper on the argument matrix),
`prevViewProjectionMatrix` @64, `currViewProjectionMatrix` @128 (current = viewToClip combined with worldToView by the
effect, which keeps last frame's copy in `prevViewProjectionMatrix`), `cameraPos` float4 @192 (the float4 argument, from the
effect's worldToView; exact derivation not checked), `widthRcp` @208 = 1/W, `heightRcp` @212 = 1/H, `mvScaleX` @216 =
W / motionScale.x, `mvScaleY` @220 = H / motionScale.y (W, H = the filter's size, `this+8/+0xc`).

Function constants of the denoiser kernels (name = AIR `air.function_constants`; values from the `BFNet_v1_Filter` constructor,
whose arguments come from the descriptor): 15 `is_tensor_layout_nchw` = false for the fused NHWC path, 16 `bfnet1_enabled` = true,
19 `spec_hit_dist_enabled` = the descriptor's specular hit distance flag, 20 `str_mask_enabled` = the strength-mask flag,
21 `compose_buffer_enabled` = the compose-buffer (transparency overlay) flag, 24 `new_overshoot_clamp` = false (all are `bool`;
`reproject_*`, `bilateral`, `denoiser_between_processing` and `update_vbbr` use 16, 19, 20, 21 as in each kernel's metadata).
The `BRNet_v3_Filter` constructor sets 9..0x1a from its own flags (same extraction, see the pass listing).

## Not done (needed before a port)

1. Done in `docs/metalfx-denoiser-passes.txt` (generated by `tools/dispatch_extract.py`): pipeline, texture/buffer bindings and
   dispatch shape of every pass of the eight filter methods above, and the filters' member tables (all pipelines with their
   kernel names, all intermediate textures with format and size: full-size RGBA16F/R16F/R8 history and ping-pong sets, half-size
   RG16F/R16F/RGBA16F DBF outputs, 1/8-size R16F; the `bilateral` loop runs 4 iterations with step 1<<i). The parameter structs and BFNet function constants
   are now documented above. Still to do: who fills the five tensor-stride vectors of `BRNetShaderParams`, `cameraPos`, the
   BRNet constructor's constants by name, the exact grids where the decompiler left a `?`, and the
   `BRNet_v3_Filter::encodeMid/encodePost` details (listed, not yet checked against the kernel signatures).
2. What `emit_v35_constants.dat` (156,832 B) feeds (unknown).
3. Descriptor → function-constant mapping for `MaxPoolConv`, `ConvKxKStride2`, `Conv3x3Stride1` (as `tools/bbr_params.py`
   did for the BBR convs) and the buffer layouts the cells use.
4. Translating those kernels (BBR's translator covers `Conv3x3Stride1/2U8`; the fp16 `MaxPoolConv`, `ConvKxKStride2*`,
   `Conv1x1FusedBilinear`, `BilinearUpsample2xSkipF16`, `Conv3x3SigmoidD2S` are untested) plus about ten denoiser and V3 kernels.
5. A reference output. There is none, so correctness could only be judged by eye or against a capture from a Mac.

`tools/mpsgraph.py` reads the package graphs, `tools/host_trace.py` the per-dispatch bindings (see their docstrings).

## The port in m4r: V3 stage sizes, and m4r's own passes (2026-11-20)

**V3 texture sizes.** The `BRNet_v3_Filter` textures at network resolution (temporal, the ema/variance pairs, aabbMin/Max,
largeWindowVarianceRatio, sample statistics and counters) are `outputSize / 2`, and the spatial one is `outputSize / 4`, NOT the
32-aligned UBFNet tensor size (only the tensor is padded; `brnet_v3_build_tensors_t` / `brnet_v3_buff_desc_t` describe it).
`brnetv3_anisotropic_gaussian_filter` samples all of them at `(gid + 0.5) / outputSize`, and `brnetv3_lowres_signals` computes its uv
from `bytes[6] = outputSize / 2`. With tensor-sized textures (1280x736 for 1440p) the filter read the network outputs 736/720 too far
down, and the recurrent ema/variance/temporal state drifted by that ratio every frame: a growing dark band above the gun with a still
camera and a doubled scope/gun in motion (what parked V3 in game). With the sizes fixed, V3 on the in-game captures: no halo, no ghost
after the first frames, no fireflies or puddle glitter, high-frequency noise 30-55% below the BBR path on static scenes and lower
frame-to-frame flicker in motion; brightness equal to the BBR path; `run_denoise_api` bit-identical to the offline harness.

**m4r's own passes (NOT MetalFX behaviour).** Between the denoiser and the upscaling stage (BBR or V3) the runtime plan has a block of
GLSL passes on buffer copies of the denoised colour:
- `nn_despeckle` (BBR path only; `M4R_NO_DESPECKLE=1` turns it off): caps a pixel at 1.5x its third-brightest neighbour.
- `nn_glossy` (both paths; `M4R_NO_GLOSSY_SMOOTH=1` turns it off): three a-trous steps (5x5 B-spline taps at distance 1, 2, 4)
  weighted by normal, depth and roughness similarity and by log-luminance distance to the 3x3 geometric mean, blended in only for
  low-roughness pixels (full below 0.15, none above 0.30) that move more than 1 px (full at 3 px). Why: `reproject_warp` drops the
  history of glossy pixels that move (weight `1 - clamp(|ms * flow| - 256 * rough * prevRough, 0, 1)`), by design, so moving
  reflections leave the denoiser with one frame of samples. Static pixels and rough surfaces pass through bit-exactly. Capture d
  (pan, 9 px/frame), final 1440p frame, high-pass rms of log luma on moving mirror pixels: BBR path 0.118 -> 0.029, V3 0.104 -> 0.051;
  frame-to-frame residual on moving mirror pixels: V3 0.047 -> 0.026 (BBR path 0.061). Cost at 853x480: at most 0.28 ms (every pixel
  gated in), plus three image copies.

Measuring noise: `tools/noise_report.py` (fixed 2026-11-20: its output-resolution band masks were scaled by `oh // H + 1`, i.e.
misaligned; the final-frame columns before that are wrong). Relative rms is dominated by a few emissive highlights (car lights) in the
glossy bands; log-luma high-pass on 5x5-eroded masks is more telling.

**Bicubic history warp (2026-11-20).** `brnetv3_warp_history_high_resolution` is the only kernel with a filter-mode-2 (bicubic)
sampler. The translator now renames those sample calls to `air.sample_texture_2d_bicubic.*` and lowers them to Catmull-Rom built from
9 linear taps (`air/intrinsics.py` `sample_tex_bicubic`; `M4R_LINEAR_BICUBIC=1` at plan-generation time restores plain linear).
Apple's exact cubic kernel is undocumented: Catmull-Rom is m4r's choice. Capture j (1280x720 pan): rough-surface gradient energy
0.038 -> 0.041 (BBR path 0.058), flicker and brightness unchanged, warp pass 0.14 -> 0.16 ms. V3 remains softer than BBR in motion;
with a still camera it is about as sharp (capture h steady state: 0.051 vs 0.056).
