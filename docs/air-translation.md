# AIR → AMDGPU translation

MetalFX's GPU work is ordinary Metal compute kernels in `default.metallib`. m4r compiles those kernels, as
shipped, for RDNA3.

## Reading the metallib

`air/metallib.py` parses the MTLB container: a header with the offsets of the function list, the public and
private metadata and the bitcode section; per function a tagged entry (`NAME`, `TYPE`, `HASH`, `OFFT` with three
u64 offsets, `MDSZ` = bitcode size, ..., `ENDT`). Each function's bitcode is a complete LLVM module with
Apple's bitcode wrapper header (`0x0B17C0DE`). Upstream LLVM 23 reads all 90 modules of macOS 27.2.1
(`llvm-dis`); carving the section at `BC C0DE` magic instead of using `OFFT`/`MDSZ` gives truncated modules,
which is why earlier attempts reported "Invalid encoding".

## What the translator changes (`air/m4r_air.py`)

The kernel's own instructions are kept. The translator only:

1. **Specialises function constants.** Metal kernels declare constants (`!air.function_constants`, with names
   and indices) whose values the host supplies when it builds a pipeline. Their placeholder globals get the
   values; derived globals computed by the module's static initialiser are folded by LLVM's `globalopt`. A kernel
   whose initialiser cannot be folded is rejected.
2. **Maps address spaces.** AIR device (1) and threadgroup (3) memory are AMDGPU global (1) and LDS (3); AIR
   constant (2) becomes AMDGPU constant (4), so constant buffers are read with scalar loads; private allocas move
   to address space 5.
3. **Implements AIR built-ins** (`air/intrinsics.py`) with AMDGPU intrinsics: texture reads, writes, samples,
   gathers and atomics become `image_*` instructions on real RADV descriptors; SIMD-group operations become wave32
   lane permutes and whole-wave reductions; math and conversions become the matching LLVM operations
   (float→int conversions saturate, as Metal's do).
4. **Wraps the kernel** in an `amdgpu_kernel` that derives Metal's thread-position built-ins from the workgroup
   and workitem ids, and returns early for threads outside the grid (Metal's `dispatchThreads`).

LLVM's AMDGPU backend then compiles the module (`opt -O3`, `llc`, `ld.lld`) for gfx1101, wave32.

## Kernel argument block

The code object reads one argument block through the kernarg pointer; vkhip passes its address as the
pipeline's only push constant. The manifest (`.json`, and `.args` for the C++ runtime) lists:

- one 8-byte slot per buffer, texture and sampler argument, with its Metal binding index;
- a texture slot holds the address of a 64-byte record: the sampled-image descriptor, then the storage-image
  descriptor, both obtained from RADV with `vkGetDescriptorEXT`; a nil texture is a record of zeros;
- `u32 grid[3]` (threads per grid);
- byte offsets for threadgroup-memory arguments.

Constexpr samplers declared inside a kernel become writable globals; the runtime decodes their Metal sampler
state, creates a matching `VkSampler` and stores its descriptor in the code object's data buffer.

### Sampler state words (inferred)

From the seven distinct states MetalFX uses and the coordinates the kernels pass with them:

| bits | meaning | evidence |
|---|---|---|
| 0–2, 3–5, 6–8 | s/t/r address mode, 1 = clamp_to_edge, 0 = clamp_to_zero (assumed MSL enum order) | inferred |
| 9–10, 11–12 | mag / min filter, 0 nearest, 1 linear, 2 not supported yet (probably bicubic) | inferred |
| 15 | `coord::pixel` | kernels using it pass pixel-space coordinates, the others normalised ones |
| high 16 bits of word 0 | `0x7bff` (half max), presumably the LOD clamp | inferred |

## Fragment functions and tile kernels

- A fragment function (e.g. `copyImage`) runs as compute: `air.position` is the pixel centre, and the returned
  colour is stored into a render-target texture passed as a hidden argument (Metal binding 255 in plans).
- A tile kernel's implicit imageblock store (`warpImage`) writes the same render target at
  workgroup · tile + in-tile coordinate; m4r dispatches one threadgroup per tile.
- A store names its render target by a literal index; each index gets its own hidden texture argument:
  `__rendertarget` (plan binding 255), `__rendertarget1` (254), `__rendertarget2` (253). V3/V4's `unified_*_tile` use three.
- Imageblock struct members in the metadata carry extra tag pairs (`!"air.render_target", i32 N`); `struct_info` skips them.
- LLVM 23's gfx11 true-16-bit register mode miscompiles `findOpticalMotionVectorsV3/Ref` ("expected inline
  constant"); kernels are built with `-real-true16`.

## Not done yet

- compute-kernel imageblocks (`air.imageblock_data`, `write_imageblock_slice_to_texture_2d`): the `unified_*_tile` kernels
  use them only when the function constant `using_compute_imageblock` is set; they are stubbed and `translate()` refuses that
  setting.
- Metal's partial edge threadgroups: Metal shrinks `threads_per_threadgroup` at the grid edge; the translation
  reports the shrunk size but forms SIMD groups from the full threadgroup shape.
- Floating-point reduction order in `simd_sum` and the exact rounding of Apple's texture filtering differ from
  Apple's GPUs; results are not expected to be bit-identical to a Mac.
