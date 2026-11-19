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
- **fp16 GEMM (f16 inputs, f32 accumulators): RDNA3 WMMA** (`v_wmma_f32_16x16x16_f16`, `_fgemm_wmma`). On gfx1101 this
  instruction gives **bit-identical** results to the chain of `v_dot2_f32_f16` over k in increasing order starting from C
  (`_fgemm_dot2`, `M4R_MXU_FP16=closed`): a probe compared 768,000 results with zero and nonzero C, all equal. So C goes into
  the WMMA as its accumulator (adding it afterwards would round differently), and the wave's LDS slot transposes A/B^T, C and
  D between Apple's layout and WMMA's. Checked on every output frame of the synthetic fixture and of captured game frames.
  SBBRNet at 2560x1440 output: 2.99 -> 0.43 ms.
  Note: neither instruction is correctly rounded (`v_dot2_f32_f16` errs by up to about 2.4 ulp of the largest term against exact
  arithmetic); "exact" here means equal to the earlier lowering, which is deterministic, not equal to true math.

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

## GEMM chains in WMMA layout

`air/gemm_chain.py` (on by default, `M4R_GEMM_CHAIN=0` turns it off) keeps an accumulator in WMMA's layout while it only flows
from GEMM to GEMM: the module is optimised first with the MXU intrinsics still opaque (so the accumulator arrays are registers
and loops are phis), then each GEMM becomes `wgemm` (WMMA with C as the accumulator, as before) and conversions between the
layouts (`a2w`, `w2a`, through the wave's LDS slot) remain only where other code reads or produces an accumulator: a value
entering a loop is converted once in the block before it, a value read after a loop once in the block that reads it. Same
instructions, so the results are bit-identical (every output frame of the synthetic and the captured game fixtures). At
1280x720 -> 2560x1440: first layer 0.50 -> 0.31 ms, SBBRNet 0.43 -> 0.28, each 32-channel 3x3 convolution 0.31 -> 0.17; the
upscaler 3.79 -> 2.96 ms. Masked MXU loads and stores with a full mask are single vector accesses (MetalFX passes 15 or 0).

## Other GPU generations

Plans are built for one GPU (`install.py --gpu`, detected from the amdgpu KFD topology by default; the plan's first line names
it and the runtime refuses a plan built for another GPU, which it identifies from the chip name RADV reports). Every
generation runs the same lowering, written against RDNA3's WMMA instructions and register layout (including the GEMM chains
and the weights read straight into WMMA layout); only the two instructions are executed differently (`air/m4r_air.py
wmma_impl`, `air/intrinsics.py wmma_impl_defs`; `M4R_WMMA_IMPL` overrides):

| GPUs | how `v_wmma_*_16x16x16` runs |
|---|---|
| RDNA3, RDNA 3.5 (gfx110x, gfx115x) | natively |
| RDNA4 (gfx120x) | gfx12's WMMA: its operands hold 8 of the 16 k (half h: k 8h .. 8h + 7), so each lane passes its half of the gfx11 operand; its accumulators hold row i + 8h in VGPR i instead of row 2i + h, converted with one cross-half exchange (`v_permlanex16`) per register before and after |
| RDNA2 (gfx103x, no WMMA) | software: each accumulator element is the chain of `v_dot2c_f32_f16` (fp16) or `v_dot4_*` (int8) over k in increasing order starting from C, the A row broadcast with DPP `row_share` (half 1 first rotated by one lane with `row_ror`, so that row_share n reads row n + 1 there). One inline-asm statement per output row: otherwise LLVM merges the broadcasts of products sharing their A operand and keeps them alive across them, which spilled (and RDNA2 kernels may not use scratch, below) |

Validated on the RX 7700 XT (gfx1101), all six captured game frames bit-identical to the native build:
`M4R_WMMA_IMPL=soft` (the RDNA2 code, with gfx11's name of the same fused DPP dot instruction, `v_dot2acc_f32_f16`) and
`M4R_WMMA_IMPL=gfx12emu` (the RDNA4 conversions around an emulation of gfx12's WMMA built from gfx11's, as d4r's layout shim).
On that card the soft path takes 4.2 ms per 1440p Performance frame and the native one 2.0 ms. Not checked: whether
`v_dot2c_f32_f16` on an RDNA2 GPU rounds as RDNA3's WMMA does (the formula is the same either way), and anything on RDNA2 or
RDNA4 hardware.

vkhip (runtime/vkhip) knows each generation's encodings of the instructions it patches (gfx10.3 numbers `s_getpc_b64`,
`s_mov_b32`, `s_branch`, `s_mov_b64` differently; gfx12 keeps gfx11's) and gfx12's getpc sequence (`s_sext_i32_i16` and
`s_wait_alu` inside it). RDNA2 has no architected flat scratch, so LLVM gives every kernel the private segment buffer in
s[0:3] and the kernarg pointer in s[4:5]: vkhip copies the push constant there first (`s_mov_b64`) and refuses kernels that
use scratch. `tests/vkhip_parse` checks the parsing and patching of code objects for any generation without a GPU.
