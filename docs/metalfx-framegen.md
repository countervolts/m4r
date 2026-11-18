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

## Open

- full binding list of encodeTo, FrameGenParams / the 176-byte shader parameter block, pyramid level count
- `findOpticalMotionVectorsV3` fails in instruction selection ("expected inline constant") — translator issue
- copyImage (fragment) as compute
