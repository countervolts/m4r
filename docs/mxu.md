# Apple's MXU matrix instructions on RDNA3

MetalFX's network kernels (`Conv*`, `SBBRNet`, `MaxPoolConv`, ...) call Apple GPU intrinsics directly:

| intrinsic | operands per lane (SIMD group of 32) | meaning |
|---|---|---|
| `llvm.agx3.igemm.v8i32.i64.i64.v8i32(16,16,16, 9, A, ta, B, tb, C, 9)` | A, B: 8 × int8 packed in i64; C: 8 × i32 | D = C + A·B, 16×16×16; type 11 = u8, 75 = s8 (inferred) |
| `llvm.agx3.fgemm16x16x16.v8f16.v8f16(A, 0, B, 0, C)` | A, B: 8 × half; C: 8 × float | D = C + A·B in f32 |
| `llvm.agx3.load/store.with.emask.global.v4i8/v4i16(p, m, 15, size)` | 4 consecutive elements | element i active when bit i of `m & 15` is set; masked loads give 0 |
| `llvm.agx2.device.load.cachectl` | | plain load |

## Lane layout

Which matrix element each lane/slot holds is the hardware's convention. `tests/probe_mxu.py` measures it from
MetalFX's own `Conv3x3Stride2U8`: a probe build records the A and B operands of one SIMD group and gives each
accumulator element its id; label-encoded inputs then show the (pixel, channel), (channel, output channel) and
(pixel, output channel) of every element. The result is `air/mxu_layout.py` (three permutations of the 16×16 tile).

**Verified:** with that layout, the translated `Conv3x3Stride2U8` and `Conv3x3Stride1U8` (block-major weights)
reproduce a numpy convolution exactly on random data (`tests/verify_conv.py`, max difference 0).

**Not verified:** `fgemm16x16x16` is assumed to use the same layout (it is only used by the first layer's
`Conv3x3Stride1Enc0FQSplit` and by `SBBRNet`); the s8 variant (type 75) is not exercised by the BBR network.

## Lowering

Apple's lane layout (measured, `air/mxu_layout.py`) has a closed form in the lane bits, which the lowering uses instead of table
lookups (`_mxu_closed_forms_ok` checks it against the tables every time): a lane holds, for A and for the accumulator, 4
consecutive columns/k of rows `r` and `r + 8`, and for B^T, 4 consecutive columns at `k` and `k + 8`, with
`r = lane bits (1, 2, 4) -> (bit 1 of lane, bit 2, bit 4)` and `k0 / c0 = 4 * lane[0] + 8 * lane[3]`.

- **Integer GEMM (u8/s8, i32 accumulators): RDNA3 WMMA** (`v_wmma_i32_16x16x16_iu8`). WMMA wants lane `l` to hold row or column
  `l & 15` with all 16 k, so A (two dword stores) and B^T (eight byte stores) are scattered through the wave's LDS slot, read
  back as two 16-byte rows, multiplied, and the result written to LDS row-major and read back in Apple's layout (two 16-byte
  loads) and added to the accumulator. Integer sums do not depend on the order, so this is **bit-identical** to the dot-product
  lowering it replaced (`M4R_MXU_INT=dot` builds that one); checked on the numpy convolution (`tests/verify_conv.py`) and on
  every output frame of the full pipeline. The integer convolutions went from 3.3 ms to 0.45 ms per frame.
- **fp16 GEMM (f16 inputs, f32 accumulators): a chain of `v_dot2_f32_f16`** over k in increasing order, as before, but with
  the closed-form layout: each lane loads its 2 rows of A and 4 columns of B once instead of 32 operand vectors and 8 table
  lookups. Same arithmetic, so bit-identical to the table version (`M4R_MXU_FP16=table`); `SBBRNet` 1.16 → 0.57 ms, the first
  layer 0.32 → 0.14 ms. **WMMA is not used here on purpose:** `v_wmma_f32_16x16x16_f16` adds the products in an order and with
  roundings the ISA does not specify, so its f32 sums would differ from the dot2 chain in the last bits, which is a change of
  the numerics, not only of speed.

Whole frame (640x360 → 1280x720, RX 7700 XT): 5.3 ms before, 1.7 ms now (`M4R_PASS_TIME=1 build/run_bbr ...` prints the time of
every pass).

## LDS slots

Each wave scatters its operands into its own slot of one LDS array (`__m4r_mxu`), and every GEMM kind uses the same slot
size: 1024 bytes (an fp16 tile needs A and B^T at 256 halves each), with one slot per wave of the threadgroup.

Slots used to be sized per kind (512 bytes for integer GEMMs, 1024 for fp16). A kernel that mixes both, SBBRNet (layer 0 is
integer, the rest fp16), then had a wave in an fp16 layer overlapping the next wave's integer slot. While the waves of a
threadgroup ran in lockstep nothing happened, so small dispatches looked fine (up to about 80 threadgroups of SBBRNet);
with more contention the waves drifted apart and the output became random: 15% of the bytes of the sBBRNet output wrong, the
final image 28.0 to 28.2 dB from run to run. The conv kernels use integer GEMMs only and were never affected.

How it was found, in case something similar shows up: `M4R_PASS_HASH=1 build/run_bbr <plan> <fixtures> 1 -1` submits one
pass at a time and prints a hash of every buffer and texture that changed, so two runs show the first pass that differs
(SBBRNet). Dispatching that pass alone, in 8-group ranges, in chunks, with extra LDS, with all registers pre-filled with junk,
with padded buffers and with zeroed LDS tiles ruled out, in turn, scratch memory (`tests/test_scratch.cpp` with
`tests/scratch_test.ll` shows that private memory works in injected pipelines with 14,400 waves), shared-memory size, undefined
values, stale registers and out-of-bounds access. `tests/check_determinism.sh <plan> <fixtures>` is the regression check.

## vkhip note

vkhip's donor shader indexed its shared array with the thread id, so ACO allocated only threadgroup-size × 4 bytes
of LDS; kernels with more LDS than that (the GEMM tiles here) overflowed into other waves. The donor now indexes
with a value the compiler cannot bound.
