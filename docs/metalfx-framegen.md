# MetalFX frame interpolation — what is known

Static reading of macOS 27.2.1 MetalFX: `_MFXFrameInterpolatorEffect` drives a C++ `FrameGenImpl<MFXDevice3>`
(constructor builds the pipelines, `encodeTo(device, commandBuffer, FrameGenParams const&)` encodes one interpolated
frame). Classical (no neural network): optical flow on a luminance pyramid, combined with the game's motion vectors,
then a warp of the previous and current frames. Inferred from code and names unless marked verified.

## Pipelines (constructor; slot = offset in FrameGenImpl)

Function constants are set on one `MTLFunctionConstantValues` object that is reused, so later pipelines inherit
earlier settings.

| slot | kernel | constants |
|---|---|---|
| 0x00, 0x08, 0x10 | computeLuminance | (3,4) = (0,0), (1,0), (0,1); index 1 set first |
| 0x18 | analyzeHistogram | |
| 0x20 | downSample | |
| 0x28 | generateLuminancePyramid | |
| 0x30 | findOpticalMotionVectorsV3 | |
| 0x38 | splatOpticalMotionVectors | |
| 0x40 | initMotionVectors | |
| 0x48 | clearAppMotionVectors | |
| 0x50 | splatAppMotionVectors | |
| 0x58, 0x60 | dilateDepth | index 2 = 0 / 1 |
| 0x68, 0x70 | dilateDepth | index 5 = 1, index 2 = 0 / 1 |
| 0x78, 0x88 | upsampleAppMotionVectors | without / with index 5 |
| 0x80, 0x90 | fillCracksAppMotionVectors | without / with index 5 |
| 0x98 | inpaintMotionVectors | |
| 0xa0 | medianFilter | |
| 0xa8 | debugShader | |
| 0xb0, 0xb8, 0xc0 | warpImage (tile shader) | useDistortionfield/writeUndistorted = (0,0), (1,1), (0,1) |
| 0xc8, 0xd0, 0xd8 | warpImage (tile shader) | as above with useDisoccBlend = 1 |
| 0xe0 | copyImage (fragment) | |

`warpImage` is a tile kernel: one thread per pixel of a tile, threadgroup memory and barriers, result stored to the
implicit imageblock (the render pass's colour attachment). m4r runs it as a compute kernel with one threadgroup per
tile and lowers the imageblock store to an image store into the render target.

## encodeTo pass order (debug group names from the code)

`MTLFX_Frame_Interpolation`: "Downsample" (computeLuminance variant, downSample), "Analyze histogram", "Optical
Motion Vectors" (findOpticalMotionVectorsV3 per pyramid level, medianFilter), "clear temp motion vectors"
(initMotionVectors), "Splat optical motion vectors", "Inpaint optical motion vectors", "clear game motion
vectors", then the game-motion path (splat/upsample/fill cracks), depth dilation, warp (`reproject_warp` /
`reproject_blend` names appear in strings) and the UI composite (copyImage). Environment:
`MTLFX_FRC_DISOCC_BLEND` selects the disocclusion-blend warp variants.

## encodeTo bindings (first reading; fg = FrameGenImpl, P = FrameGenParams)

Compute encoder, with the 176-byte shader parameter block (`FrameGenShaderParams`, field names from the kernels'
metadata) as transient buffer 0:
1. Downsample: computeLuminance (slot 0x0/0x8/0x10 by flags): tex0 current colour, tex1 ?, tex2 ?, tex3 fg+0x220
   (when flagged), buf1 fg+0x230 (histogram); then downSample (0x20) once per pyramid level: tex0 level i, tex1 i+1.
2. Analyze histogram (0x18): buf1/buf2 fg+0x230, buf3 fg+0x230 + 0x4800; threadgroups {1,9?} x {256}.
3. Optical Motion Vectors: tex4 = P+0x18; per level: findOpticalMotionVectorsV3 (0x30, tex0-3, threadgroups) and
   medianFilter (0xa0, tex0-1).
4. clear temp motion vectors: initMotionVectors (0x40): tex0 fg+0xe8, tex1 fg+0xf0.
5. Splat optical motion vectors (0x38): tex0 fg+0xe8, tex1 fg+0xf0, tex2, tex3, tex4 fg+0x1b0, tex5 fg+0x1e8.
6. Inpaint optical motion vectors (0x98): tex0 fg+0x218, tex1 fg+0xe8, tex2 fg+0xf0.
7. clear game motion vectors (0x48): tex0 fg+0xf8, tex1 fg+0x100.
8. dilate depth [& motion vectors] (dilateDepth variant): tex0 P+0x10 (depth), tex1, tex2, tex3 fg+0x200, tex4 fg+0x210.
9. Splat game motion vectors (0x50): tex0 fg+0xf8, tex1 fg+0x100, tex2 current colour, tex3 P+0x8, tex4 P+0x0, tex5, tex6.
10. Upsample game motion vectors (0x78/0x88): tex0 fg+0x200, tex1 fg+0xf8, tex2 fg+0x100, tex3 fg+0x208, tex4, tex5.

Render pass `MTLFX_Frame_Interpolation_Fragment` (tile shaders): tile buffer 0 = shader params, tile buffer 1 =
fg+0x230 + 0x4800 (`frameChange`, from the histogram analysis), threadgroup scratch; tile textures by warpImage's
locations: 1 fg+0x208 (maskTex), 2 prevDilatedDepthW, 3 fg+0x220 (currentTex), 4 fg+0x228 (prevTex), 5 fg+0x200
(gameMotionVectors), 6 fg+0x218 (opticalMotionVectors), 7 backbuffer, 8 uiTexture, 9 distortionField, 10 current
colour, 11 curDilatedDepthW, 14 fg+0x210 (moverMaskW); then "Warp image" (warpImage variant) or "Copy image"
(copyImage), dispatchThreadsPerTile. Stub identification: 0x258015ea0 setRenderPipelineState,
0x258043f80 setTileTexture:atIndex:, 0x258043f40 setTileBuffer:offset:atIndex:, 0x258043300 dispatchThreadsPerTile.

Shader parameters computed in encodeTo (from P): motionVectorScale = P.mvScale · content / P dims, depthScale /
depthBias = reversed ? (1, 0) : (−1, 1), deltaTime = P+0x58, prevDeltaTime = fg+0x244, proj33 = n/(n−f),
proj34 = −n·f/(n−f) (n, f = P+0x5c, P+0x60), clipPrevFromClipCur from the current (P+0x70..0xef) and stored
previous (fg+0x250..0x2cf) matrices when both exist (P+0xf0, fg+0x2d0), rejectionThreshold = content w·h·80/900.

## Resources (constructor; O = output size, I = input size: `FrameGenImpl(dev, lib, O.w, O.h, I.w, I.h, colourFormat, ...)`)

levels = min(floor(log2(min(O.w, O.h))) − 2, 7) (fg+0x240); prevDeltaTime starts at 1/60 (fg+0x244); stored previous
camera matrices start as identity (fg+0x250..0x2cf) with fg+0x2d0 = "have previous" = 0. Function constant 1 is
false for a few colour formats (0x24a, 0x27e, 0x27f and a small set around 0x46) and true otherwise.

| slot | Metal format | size |
|---|---|---|
| +0xe8, +0xf0 | R32Uint | O / 8 (optical splat targets) |
| +0xf8, +0x100 | R32Uint | I (game-motion splat targets) |
| +0x1f0, +0x1f8 | R32Float | I (dilated depth, two frames) |
| +0x108 + 16·l + 8·k | R8Unorm | O >> l, l < levels, two pyramids (k = frame parity) |
| +0x178[l], +0x1b0[l] | RG16Float | O >> (l + 3) (flow per level, two sets) |
| +0x1e8 | RG16Float | last level size |
| +0x200 | RG16Float | O (game motion, upsampled) |
| +0x208 | R8Uint | O (mask) |
| +0x210 | R16Float | I (mover mask) |
| +0x218 | RG16Float | O (optical motion) |
| +0x220, +0x228 | colour format | current / previous colour copies |
| +0x230 | buffer, 18,436 bytes | histogram; +0x4800 = frame-change word |

## End-to-end result

`tools/prepare_fg.py` + `tests/run_fg.cpp` on the moving-camera fixture (`tests/scene_fg.py`, output 1280x720,
input 640x360, a moving sphere): interpolated frames score **31.9–32.1 dB** sRGB PSNR against the true in-between
frames (blending the two frames: 22.3 dB, repeating the previous one: 20.2 dB), no NaNs, 0.87 ms per frame on an
RX 7700 XT.

Tile emulation detail (verified by the result): `warpImage` rebuilds its pixel as
x = (xy.x & ~7) + lane / 4, y = (xy.y & ~3) + lane % 4, i.e. it relies on Apple forming each SIMD group of a tile
dispatch from an 8x4 pixel block (its shuffles with lane masks 1/4 and 2/8 are vertical/horizontal neighbours). m4r
runs a 32x32 tile as a 1024-thread workgroup whose wave w covers 8x4 block w; with plain row-major waves the output
is noise.

## Open

- full binding list of encodeTo, FrameGenParams / the 176-byte shader parameter block, pyramid level count
- the copy path (scene cut, `copyImage`), distortion field, UI composition, camera-matrix reprojection and the
  disocclusion-blend variants are not exercised by the fixture yet
