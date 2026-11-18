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

## Open

- full binding list of encodeTo, FrameGenParams / the 176-byte shader parameter block, pyramid level count
- `findOpticalMotionVectorsV3` fails in instruction selection ("expected inline constant") — translator issue
- copyImage (fragment) as compute
