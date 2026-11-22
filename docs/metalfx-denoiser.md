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
- `nn_despeckle` (both paths since 2026-11-20; before V3 it cut the grain V3 leaves on puddles by 26% on capture m without losing detail; `M4R_NO_DESPECKLE=1` turns it off): caps a pixel at 1.5x its third-brightest neighbour.
- `nn_glossy` (both paths; OPT-IN since the first in-game test, `M4R_GLOSSY_SMOOTH=1`: it flattened moving reflections into plastic-looking patches and erased grout lines inside them): three a-trous steps (5x5 B-spline taps at distance 1, 2, 4)
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

**Hybrid output (opt-in `M4R_HYBRID=1`, not MetalFX; `shaders/nn_hybrid.comp`).** V3 renders into its own texture, BBR upscales the
same denoised colour into the game's output, and BBR is clamped per channel to V3's 5x5 range widened by 25%. Capture j: nearly BBR's
sharpness (gradient 0.056 vs BBR 0.058, V3 0.041) with fewer fireflies. Capture k (fast pan over a wet floor): mirror-pixel flicker as
high as BBR (0.127 vs 0.124; V3 0.042) and more fireflies than BBR (its BBR input skips the despeckle clamp), so it is not the default.
Costs both upscalers (~+2 ms) and two output-size buffers in every dn3 plan.

**Speed (2026-11-20, RX 7700 XT, 1280x720 -> 2560x1440, V3 path, offline harness):** 5.86 -> 5.46 ms per frame, outputs
bit-identical. `nn_d2s` one thread per output pixel (0.29 -> 0.11 ms); WMMA convs at most 2 channel tiles per wave above 40k pixels
(4 made the 320x180 64->64 layers 2x slower; convs 1.33 -> 1.09 ms). Left as they are: Apple's `bilateral` x4 (2.0 ms; 173 VGPRs,
fewer spills to scratch, which the runtime does not support, and LLVM's unrolling is faster than not unrolling; about 70% of its
memory-bandwidth bound), the anisotropic filter (0.4 ms) and the other Apple kernels (49-83 VGPRs, texture-bound).

**Reflection-lag switch (opt-in `M4R_REFLECTION_LAG_FIX=1`, or `=<cap>`; not MetalFX; `shaders/nn_mircap.comp`).** Walking makes
reflections trail: they move differently from the surface, the game gives surface motion vectors only, and V3 blends reflection
history from the wrong place. The pass caps UBFNet's temporal history weight (channels 3..6) on low-roughness pixels (full below 0.08,
none above 0.2) by `mix(1, cap, gate)`, default cap 0.3, gate from the camera's translation per frame (smoothstep 0.003..0.02 m;
walking in Cyberpunk is 0.03-0.05 m/frame, turning or standing <= 0.001), so turning the camera is untouched (capture d bit-identical
with the switch on). Capture m (walking): ghosted tile lines across reflections fade, puddle grain 0.074 -> 0.089 (cap 0.3) /
0.119 (cap 0.01).

**Hybrid v2 (opt-in `M4R_HYBRID=<strength 0..1>`; `M4R_NO_V3=1` = BBR alone; not MetalFX).** Briefly the default; in game it shimmered on
rough surfaces: on a still capture (n) rough-surface frame-to-frame change V3 0.041, hybrid 0.060 (0.5: 0.048), BBR 0.068, while
sharpness went 0.078 / 0.100 (0.5: 0.083) / 0.105; tighter clamps (3x3, no widening) traded the two the same way, so the user picks. Both upscalers
run on the same denoised colour; `shaders/nn_hybrid.comp` takes BBR (clamped to V3's 5x5 range +25%) on rough surfaces and V3 on
shiny ones, blended by roughness (3x3 render-pixel average; full V3 below 0.10, full BBR above 0.25). Captures j / k / m (720p):
rough-surface gradient energy V3 0.039 / 0.059 / 0.040, BBR 0.054 / 0.089 / 0.055, hybrid 0.052 / 0.085 / 0.053; fireflies V3 0 / 16 / 0,
BBR 330 / 2106 / 362, hybrid 2 / 57 / 0; frame-to-frame residual on moving shiny pixels equal to V3 (0.038 / 0.072 / 0.099; BBR
0.093 / 0.169 / 0.210); brightness within 1%; no visible seam at puddle edges. Costs the BBR upscaler on top of V3 (~2 ms). If a game
gives roughness as its own texture (not normals.w) the roughness buffer stays zero and the blend is V3 everywhere.

**Albedo detail (default with V3 since 2026-11-20; `M4R_NO_ALBEDO_DETAIL=1` = off, off with `M4R_HYBRID`; not MetalFX).** V3
denoises cleanly but softens texture in motion, and every way of pairing it with a sharper upscaler traded that for shimmer (the
upscalers need the per-frame detail the denoiser averages away). Most of the softened detail is texture, which the game also gives
noise-free as albedo, the input MetalFX's BBR upscaler is built for. So: V3 writes its image into its own texture; `nn_albsum` sums
the game's diffuse + specular albedo (RGBA8) into RGBA16F; a second BBR instance (own history) upscales it; `nn_albdetail` multiplies
V3's image by that albedo's local detail ratio, `clamp(a / gaussian_sigma1(a), 0.7, 1.4)` on luminance, with Cyberpunk's
water/puddle placeholder albedo (diffuse and specular 1) excluded from the blur and the ratio faded to 1 next to it, so reflections
stay V3's. Captures m / n / k (720p walk, 480p still, 720p pan), rough-surface gradient energy V3 0.040 / 0.078 / 0.059, BBR
0.055 / 0.105 / 0.089, albedo detail 0.056 / 0.099 / 0.083; frame-to-frame change V3 0.027 / 0.037 / 0.027, BBR 0.043 / 0.066 / 0.047,
albedo detail 0.029 / 0.041 / 0.031; shiny/puddle pixels unchanged. Costs the BBR upscaler (~2 ms at 1440p) plus copies.
Things tried before it that did not help and why (2026-11-20): DLSS K (d4r) on the first-stage denoiser output (speckles, shimmer)
or on V3 run at 1:1 (softest of all); V3 1:1 into BBR (same); repeated first-stage passes (<1% fewer speckles); RCAS sharpening
(rejected); injecting reflection motion vectors for puddles (V3's lag did not follow them); physically plausible puddle albedo
(the first stage then smears reflections and brightens them ~45%).

## Denoiser speed (2026-11-20)

Measured on Alan Wake 2 frames (resampled to each mode's render size), 2560x1440 output, RX 7700 XT, GPU time of m4r_denoise (Vulkan timestamps
inside its own command buffer, `M4R_TIME=<file>`), K mode (the caller upscales), game closed.

| mode | render | original (4 passes, strict) | default now (3 passes, strict) | `FastMath` (opt-in) | `FastMath` + `M4R_ATROUS_ITERS=2` |
|---|---|---:|---:|---:|---:|
| DLAA | 2560x1440 | 17.89 ms | 13.66 ms | 10.43 ms | 9.57 ms |
| Quality | 1706x960 | 7.26 ms | 5.63 ms | 4.18 ms | 3.79 ms |
| Balanced | 1505x847 | 5.58 ms | 4.30 ms | 3.18 ms | 2.89 ms |
| Performance | 1280x720 | 4.03 ms | 3.10 ms | 2.29 ms | 2.07 ms |
| Ultra Performance | 853x480 | 1.87 ms | 1.41 ms | 1.05 ms | 0.95 ms |

Exact (bit-identical) speedups in the default (latest first): the stages around the denoiser that are not MetalFX's (roughness, exposure
scaling, depth conversion, mirror gate, despeckle, exposure undo) are OpenCL kernels that read and write the game's images directly
(`shaders/nn_aux.cl`: PREP before the Apple chain, TAIL1 and TAIL2 after it) instead of staging everything through buffer copies, which removed about
15 dispatches and copies per frame (Performance -0.26 ms, 8%; on Cyberpunk the mirror gate differs from the old shader by 1 fp16 ulp in ~10 pixels per
frame, none on Alan Wake 2); the DBF convolutions split their output channels over waves from 10000 output pixels (was 40000: Ultra Performance -5%,
Quality -3%; `M4R_NT_PX`); and three bilateral passes instead of four, `serial` plans (the dependency-level schedule measured
1.5-2.5% slower than plan order in every mode; plan line `serial`, `M4R_LEVEL_SCHEDULE=1` restores levels), and `--amdgpu-sched-strategy=iterative-ilp`
for the bilateral kernel (9% fewer vector instructions; code generation only). The remaining per-dispatch overhead is about 15 us over ~50 dispatches.

**FastMath (opt-in: `[MetalFX] FastMath=true`, or `M4R_FAST_MATH=1`).** Apple's bilateral filter calls Metal's `fast_exp`, `fast_log`,
`fast_sqrt` and `fast_pow`, which are the raw hardware instructions (v_exp_f32 and friends). The translator's default lowering is
LLVM's strict one (correctly rounded sqrt, extended-precision exp/log, range and denormal scaling around exp2/log2), which is more
accurate than Metal's and about twice as slow here: the filter goes from 0.25 to 0.13 ms per pass at 1280x720 (`air/intrinsics.py`
`fast_raw`). Runtime plans hold both bilateral kernels (the other kernels moved by under 2% and stay strict); the default is the strict
one and is byte-identical to what ran before. In d4r's terms (docs/approximations.md there) this is not an approximation of the model
(same formula and format, the instruction Apple's source asks for) but it is not bit-exact either; d4r also keeps its LLVM fast-math
switch opt-in.

FastMath also rewrites `fast_pow(x, 2)` to a multiplication and `fast_pow(2, x)` to an exp2 at the call site (the lowering is written once per
function and cannot see the constant; air/m4r_air.py, plan-time, fast variant only): 25 fewer exp and log per pixel, -1.7%, no change in the output.

As displayed (8-bit, tonemapped), fast against strict differs in 0.03-0.2% of the values, never by more than 1 level of 255, PSNR 75-83 dB
(90-120 dB in linear light).

How different: one bilateral pass on identical inputs is 99.99% bit-identical, at most 1 fp16 ulp, mean relative difference 1e-7. The
temporal history recirculates the tiny differences into a steady floor: after a few frames of the recurrent pipeline 92% of the denoised
fp16 values are identical, 98% within 1 ulp, mean relative difference 7e-5, 99.9th percentile 0.2%, worst pixel about 1%; flat over a
24-frame replay (no drift).

Tried without a gain (all exact, reverted or not kept): lowering the bilateral kernel's nearest-filter samples to clamped texel fetches
(15 -> 6 sampler samples, 0% faster), compute-only barriers between passes (0%), llc scheduler options on the bilateral kernel (best 2.3%),
true 16-bit registers (+1%, slower), wave64 (the kernel is written for 32-wide simdgroups and does not compile as wave64). Remaining cost
is the DBF network (about 0.8 ms of 21 small dispatches at 720p), the three bilateral passes, and per-dispatch overhead (about 0.8 ms
between the sum of isolated kernel times and the whole run). The aux stages (despeckle, mirror gate, exposure scaling) cost 0.3 ms in total.

**Stage profiler (`M4R_STAGE_PROFILE=<file>`).** m4r_denoise writes a GPU timestamp after every pass and after each copy section (a serial plan, so the
time between two stamps is the barrier, ramp and work of the stage that ended at the second); the mean per stage over ~295 frames is written at call
300. It adds about 0.17 ms at Performance (0.06 at Ultra Performance, about 1% at DLAA). Isolated per-pass timing undercounts: inside the real run the
strict bilateral filter costs about 0.49 ms per pass at 1280x720, not the 0.25 ms an isolated run shows.

Per-stage time at 2560x1440 output (Alan Wake 2 frames, K mode), default (3 bilateral passes, strict math), ms:

| stage | DLAA | Quality | Balanced | Performance | Ultra Perf |
|---|---:|---:|---:|---:|---:|
| Input prep (roughness, exposure, depth conversion) | 1.06 | 0.35 | 0.27 | 0.18 | 0.09 |
| Guide signals (dbfnetv1_lowres_signals) | 1.05 | 0.54 | 0.40 | 0.25 | 0.08 |
| Flow chain (dilate, splat, diff) | 0.33 | 0.18 | 0.17 | 0.16 | 0.07 |
| DBF network (21 dispatches) | 3.93 | 1.55 | 1.24 | 0.92 | 0.48 |
| Temporal reprojection (warp, blend) | 1.41 | 0.61 | 0.45 | 0.32 | 0.15 |
| Bilateral filter (3 passes) | 5.86 | 2.61 | 2.04 | 1.47 | 0.67 |
| Remodulation | 0.51 | 0.14 | 0.11 | 0.06 | 0.02 |
| Mirror gate | 0.63 | 0.15 | 0.12 | 0.09 | 0.04 |
| Despeckle | 0.62 | 0.10 | 0.08 | 0.06 | 0.03 |
| Exposure undo | 0.28 | 0.06 | 0.05 | 0.04 | 0.02 |
| Hand-off copy to the upscaler | 0.14 | 0.03 | 0.02 | 0.02 | 0.01 |
| total (profiled) | 15.82 | 6.31 | 4.93 | 3.56 | 1.66 |

With `FastMath`, the bilateral filter drops to 2.63 / 1.16 / 0.91 / 0.66 / 0.31 ms and the other rows do not move (totals 12.57 / 4.87 / 3.80 / 2.75 / 1.30).

## GPUs without WMMA: RDNA2 (2026-11-21)

The network kernels (`shaders/nn_cell.cl`, `nn_conv_wmma.cl`) use gfx11's `v_wmma_f32_16x16x16_f16`. `tools/denoise_nets.py nn_impl()`
picks how the convolutions run: `wmma` on RDNA3 / 3.5, `dot` everywhere else, `M4R_NN_IMPL=wmma|soft|dot` at plan time overrides.
All three give the same bits: WMMA equals the chain of `v_dot2_f32_f16` over k in increasing order starting from C (docs/mxu.md), and
both other paths compute exactly that chain per accumulator (zero-padded input channels skipped: their products are exact zeros).

- `dot` (default without WMMA): lane = pixel, weights the same for every lane, so they are scalar loads and `v_dot2c_f32_f16` reads
  them from SGPRs (`shaders/nn_dot.h DOT_BLOCK`; `nn_cell.cl` with `DOT=1`, `nn_conv_dot.cl`). The limit is the weight stream through
  the scalar cache (a 64->64 3x3 layer is 73 KB, the scalar cache 16 KB; scalar loads complete out of order, so only one chunk can be
  in flight): each lane computes MPX pixels (rows) so every weight load serves MPX of them (4 on the 64-channel cells, 2 elsewhere),
  at most 64 sums per lane (~91 VGPRs) by splitting output channels over WC waves (8 on cells below 20000 pixels), chunks of 4
  channels. The UBF head (44 channels) writes through shared memory (lane-per-pixel 2-byte stores 88 bytes apart did not coalesce).
  Compiler traps, all handled in `nn_dot.h`: LLVM hoists every scalar load of an unrolled block to its top (SGPR spills -> scratch;
  scheduling barriers + an opaque asm zero per chunk address against the load-store vectorizer), sinks the last products into the
  guarded store block (`PIN`), and unrolls the tile fill (`FILL_LOOP`, DOT only).
- `soft`: the WMMA kernels with each product as 64 `v_dot2c_f32_f16_dpp row_share` (`wmma_soft`, as the AIR translator's RDNA2 path;
  A loaded with the upper half-wave rotated by one row). Slower than `dot` here; kept for A/B on RDNA2 hardware.

Speed on the RX 7700 XT, Cyberpunk 1280x720 -> 2560x1440, DBF cells / UBF convs (ms): WMMA 0.38 / 0.35; `dot` 0.70 / 0.39, and
0.80 / 0.40 with VOPD off (`M4R_LLC_FLAGS=--amdgpu-enable-vopd=0`: RDNA2 cannot dual-issue, so this is the proxy for RDNA2's issue
rate; ~1.6x WMMA, under the 2x of WMMA's fp16 rate over plain dot2); `soft` 1.45 / 0.67. Bit-identical to the WMMA plans on captures
cp2077-m (720p) and cp2077-n (853x480), offline (dbf_feat/noise/tw, denoised, v3out, 6 frames) and through `run_denoise_api`.

Also needed for RDNA2 at all: three of Apple's kernels (`luma_log_sum_denoiser`, `reproject_warp`, `dbfnetv1_lowres_signals`) had an
8-byte private segment that no instruction uses (InstCombine turns any()/all() on vector compares into `bitcast <N x i1> to iN`, which
the backend lowers through a stack temporary); vkhip refuses scratch on RDNA2. `air/m4r_air.py expand_bool_vector_bitcasts` rewrites
those bitcasts after `opt` (same value). The gfx1030 1280x720 -> 2560x1440 runtime plan: 45 kernels, none with scratch, all parse in
`tests/vkhip_parse`. Not run on RDNA2 hardware.

Inline-asm pitfall (found here, also fixed in `air/intrinsics.py _wmma_soft`): asm whose outputs are written before all inputs are read
needs early-clobber outputs (`=&v` / `+&v`); otherwise LLVM may put an input and an output with the same value (zero padding and a
zero start) in one register. It zeroed outputs of UBF's first layer in the soft path; the AIR path's outputs did not change on the BBR
motion fixture (8 frames, soft = native), but its code did in 10 int8 kernels.

## Non-unit normals threw the history away (Control, 2026-11-21)

Symptom: in Control the MetalFX denoiser stayed nearly as noisy as a single frame. Offline (capture `control-b`, 1280x720, camera
still) the per-pixel running count `reproject_warp` keeps in `running_mean.z` had a median of 1.9 after 12 frames and 47% of the
pixels never got past 1.5: the history was rejected almost everywhere, every frame.

Cause: about 40% of Control's normals (`R8G8B8A8_SNORM`, world space, roughness in w) are not unit length (0.72-0.78). Each history
tap in `reproject_warp` is weighted by `pow(max(dot(n, n_prev), 0), 128)` (with the bilinear, flow-difference, motion/roughness,
roughness-difference and demodulation-Y terms) and dropped below 0.01; the kernel normalises the normal only when its length is in
(0.99, 1.005) and stores it as it is otherwise, so two 0.75-long normals give at most 0.56^128 = 0. Alan Wake 2 and Cyberpunk give
unit normals (RGBA16F), so they never hit this. (Not a cause, for the record: `fc` stays at 1 by design: `reproject_warp` writes the
weighted mean of the previous `fc`, `reproject_blend` adds `clamp(sigmoid(tw) + valid * boost, 0, 0.95)`, and DBFNet's temporal
weight logits are about -21..-26 on every capture; the accumulation runs through the running count and the boost.)

Fix (m4r, not MetalFX): `nn_aux.cl` LOWRES normalises normals whose length is outside (0.99, 1.005) (zero normals stay zero) into the
plan texture `nrm_fix`, which `reproject_warp` (tex 3) and the network input read instead of the game's normals. Normals in that range
pass through unchanged, so unit-normal games are byte-identical (aw2-c: 48/48 outputs). Control: running count median 1.9 -> 11.6 of
12, pixels without history 47% -> 4%, flat-region noise of the denoised image -35% (0.081-0.103 -> 0.047-0.074 per frame; same at
1505x847 in DLSS K mode), texture detail unchanged; +0.1 ms at 720p. Hair keeps some speckle: 16% of its pixels still fail the normal
test because strand normals change with the jitter (MetalFX's threshold, left as is). d4r's FSR RR path already normalises
(`fsr_rr_prepare`), so its remaining noise has another cause (Control gives no hit distances).

## Flicker on characters: still accumulation and jitter-proof normal test (Control, 2026-11-21)

Capture `control-c` (853x480, camera still, the player character in view): after the normal fix the character still boiled (blotchy
shading noise moving every frame). Two causes, two m4r changes (not MetalFX):

1. **History test vs normal maps.** At low render sizes the game's normal maps change with the jitter: on a still camera 26% of the
   character's and 52% of the floor's pixels failed `reproject_warp`'s `pow(dot, 128)` test. LOWRES (NSMOOTH, default; plan-time
   `M4R_PLAN_NO_NSMOOTH=1` off) now gives `reproject_warp` (tex 3, `nrm_fix`) the depth-aware 3x3 mean of the unit normals (neighbours
   within 2% of the depth), and the bilateral passes a separate per-pixel normal + roughness (`nrm_px`, instead of reproject_warp's
   `normal_roughness_tex`): a first version that let the bilateral passes use the smoothed normal blurred normal-mapped detail (floor
   plates; detail metric 0.34 -> 0.15). With the split, every pixel keeps its history and detail is unchanged; flicker barely moves (the
   history was not the main cause).
2. **MetalFX's history cap.** `reproject_warp` caps the history at 0.8 per frame (about a 5-9 frame average), which leaves coarse shading
   noise boiling on large soft surfaces. TAILF flag 128 (runtime, while the camera is still; `M4R_NO_STILL_ACC=1` off, weight floor
   `M4R_STILL_ACC_MIN` 0.2): pixels whose motion vector is under a quarter pixel (fading out by half a pixel) average the **3x3 mean** of
   the result over frames (history in `mir_hist`, same luminance guard as the mirror accumulation) and keep the frame's own detail
   (pixel minus its 3x3 mean); the corrected pixel is clamped to its 3x3 neighbourhood of the frame. Whole-pixel accumulation was tried
   first: flicker -74% but softer textures, because the denoised image still carries the frame's jitter (the upscaler resolves it), so
   averaging pixels averages sub-pixel-shifted detail; the unclamped 3x3 version left dark dots on silhouettes. Mirror pixels (flag 64)
   keep their whole-pixel accumulation.

Result (cycled static captures, low-pass frame-to-frame flicker / detail vs the multi-frame input mean): character 0.098 -> 0.054
(-45%), whole frame 0.040 -> 0.021, detail 0.217 -> 0.207; control-b 0.036 -> 0.019, detail 0.340 -> 0.327; aw2-c 0.028 -> 0.022,
detail 0.530 -> 0.520; moving camera (aw2-d) unchanged (0.01%). Cost: LOWRES +18 loads per pixel, TAILF +1 texture.

## Floor blotches and hair speckle (Control, 2026-11-21)

- **Mirror gate range 0.07-0.10** (was 0.10-0.15; `M4R_MIRROR_R0`/`R1`). Control's glossy floor (roughness 0.12-0.15, varying per
  pixel) fell into the gate in patches; gated pixels skip MetalFX's spatial filter and kept dark ray-noise blotches (capture
  `control-d`, 853x480). Alan Wake 2's water is at 0.04-0.06, so its mirrors stay fully gated (aw2-water: 0.7% of pixels change, by
  0.01% on average); the floor now gets the normal filter (floor high-pass noise 0.097 -> 0.080, same as with the gate off).
- **Dark despeckle** (TAILF flag 256, `M4R_NO_DARK_DESPECKLE=1` off): a pixel darker than 0.75 x its second darkest neighbour (`DSK`) is
  lifted to that neighbour's level. Hair at low render sizes keeps ray dropouts because strand normals stop the bilateral filter; the
  demodulated lighting is speckled, the albedo is smooth. Pixels on thin dark lines have two dark neighbours and stay (floor plate
  ridges intact; 1-3% of pixels change, scattered single dots). Strong dark dots in hair (below half the 3x3 median): control-c
  0.031 -> 0.010, control-b 0.016 -> 0.004.

## Faint residual noise: what did not work (Control, 2026-11-21, capture `control-e`)

Remaining faint noise at 853x480 (neck/shoulder strokes, sparkles on dark floors, grain behind glass) is dense fine noise that
MetalFX's temporal blend lets through at its ~20% current-frame weight, not isolated outliers. Tried and left opt-in (default off):
- **Temporal outlier clamp** (TAILF flag 512, `M4R_TEMPORAL_CLAMP=1`; per-pixel running mean luminance in `pxh0`/`pxh1`, the pixel kept
  within [0.7, 1.4] x its 3x3 neighbours' running means while the camera is still): flicker -1..-3%, detail -1.5% (Control) / -3%
  (Alan Wake 2). Narrower ranges ([0.85, 1.15], or [0.75, 1.33] x the median in smooth neighbourhoods) dulled the floor plates'
  ridges (detail -9..-12%) and still left most of the shoulder strokes (they move from frame to frame, so the neighbours' means sink too).
- **Strict sparkle removal** (flag 1024, `M4R_SPARKLE=1`; brighter than 1.33 x the second brightest neighbour): floor sparkles 3.04% ->
  2.96% of pixels (they have bright neighbours), Alan Wake 2 detail -15% (real glints).
Removing it outright would need more temporal weight on whole pixels; the denoised image still carries the jitter, so that softens
textures (see the still accumulation above) unless the history is re-sampled to each frame's jitter (TAA-style), which also softens.
A higher render size (Performance instead of Ultra Performance: 2.25x the rays per output pixel) is the clean lever.
