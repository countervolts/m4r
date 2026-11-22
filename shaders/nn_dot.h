// m4r: dot-product helpers for the convolutions on GPUs without WMMA (shaders/nn_conv_dot.cl, shaders/nn_cell.cl with DOT=1)
// SPDX-License-Identifier: GPL-3.0-only
#ifndef M4R_NN_DOT_H
#define M4R_NN_DOT_H
typedef half half2v __attribute__((ext_vector_type(2)));

// c + a.x * w.x + a.y * w.y: v_dot2c_f32_f16 (gfx11: v_dot2acc_f32_f16), the weight word from an SGPR
static inline float dot2(uint a, uint w, float c) {
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(half2v, a), __builtin_bit_cast(half2v, w), c, false);
}

#ifndef CH
#define CH 2   // output channels per weight chunk (2: 16 dwords)
#endif
// acc[m][0..nc) += a[m][0..np) . w[c][0..np) for the mp pixels m of the lane (w: 8 dwords per output channel at wq), the weights in
// chunks of CH channels, each chunk used by all mp pixels. Chunk g + 1 is requested after the first product of chunk g: that product
// waits for chunk g (scalar loads complete out of order, so the wait covers every request before it), and chunk g + 1 lands while
// chunk g's other products run. The scheduling barriers keep that order, and an opaque zero added to each chunk's address keeps the
// load-store vectorizer from merging all chunks into loads at the top of the block (both: SGPR spills into VGPR lanes, then scratch).
#define SB() __builtin_amdgcn_sched_barrier(0)
// the sums are needed here: LLVM would otherwise sink the products into the guarded store block after them, keeping every weight chunk
// alive across (SGPR spills)
#define PIN(acc, nc, mp) do { _Pragma("unroll") for (int m_ = 0; m_ < (mp); ++m_) _Pragma("unroll") for (int c_ = 0; c_ < (nc); ++c_) \
        __asm volatile("" :: "v"(acc[m_][c_])); } while (0)
static inline int opaque_zero(void) { int z; __asm volatile("s_mov_b32 %0, 0" : "=s"(z)); return z; }
#define DOT_BLOCK(acc, a, wq, nc, np, mp) do {                                                     \
        uint wcur_[CH * 8], wnxt_[CH * 8];                                                         \
        constant const uint* p_ = (wq) + opaque_zero();                                            \
        _Pragma("unroll") for (int i_ = 0; i_ < CH * 8; ++i_) wcur_[i_] = (i_ % 8) < (np) ? p_[i_] : 0u; \
        _Pragma("unroll") for (int g_ = 0; g_ < (nc) / CH; ++g_) {                                  \
            const int c0_ = g_ * CH;                                                               \
            acc[0][c0_] = dot2(a[0][0], wcur_[0], acc[0][c0_]);                                    \
            SB();                                                                                  \
            if (g_ + 1 < (nc) / CH) {                                                              \
                constant const uint* q_ = (wq) + (c0_ + CH) * 8 + opaque_zero();                   \
                _Pragma("unroll") for (int i_ = 0; i_ < CH * 8; ++i_)                              \
                    wnxt_[i_] = (i_ % 8) < (np) ? q_[i_] : 0u;                                     \
            }                                                                                      \
            _Pragma("unroll") for (int c_ = 0; c_ < CH; ++c_)                                      \
                _Pragma("unroll") for (int j_ = 0; j_ < (np); ++j_)                                \
                    _Pragma("unroll") for (int m_ = 0; m_ < (mp); ++m_)                            \
                        if (c_ | j_ | m_)                                                          \
                            acc[m_][c0_ + c_] = dot2(a[m_][j_], wcur_[c_ * 8 + j_], acc[m_][c0_ + c_]); \
            SB();                                                                                  \
            _Pragma("unroll") for (int i_ = 0; i_ < CH * 8; ++i_) wcur_[i_] = wnxt_[i_];           \
        }                                                                                          \
    } while (0)
// ---- software WMMA (SOFTW=1): v_wmma_f32_16x16x16_f16's semantics from DPP dot products, as the AIR translator's RDNA2 path
// (air/intrinsics.py _wmma_soft): D[i] of lane l (row 2i + h, column l & 15, h = l >> 4) = C, then the chain of dot2 over the 8 k pairs.
// The A row comes from the lane that holds it with DPP row_share (free on v_dot2c), so A must be loaded ROTATED in the upper half-wave:
// lane 16 + s holds row (s + 1) & 15 (then row_share 2i reads row 2i + 1 there); B is the lane's own column, as for the hardware WMMA.
// One asm statement per product: the s_nop covers the DPP read-after-VALU-write hazard LLVM cannot see inside it; the 8 accumulators
// take turns (k pair outer), so no instruction waits for the one before it; volatile: LLVM must not sink it into a divergent block
// (DPP reads the other lanes, which must all be active). Early-clobber outputs: the sums are written while the operands are still being
// read, and LLVM would otherwise give an operand and a sum with the same value (zero padding, zero start) one register.
#if defined(__GFX10__)
#define DOT2DPP "v_dot2c_f32_f16_dpp"
#else
#define DOT2DPP "v_dot2acc_f32_f16_dpp"
#endif
typedef float wsoft_f8 __attribute__((ext_vector_type(8)));
typedef half wsoft_h16 __attribute__((ext_vector_type(16)));
typedef uint wsoft_u8 __attribute__((ext_vector_type(8)));
static inline wsoft_f8 wmma_soft(wsoft_h16 a, wsoft_h16 b, wsoft_f8 c) {
    const wsoft_u8 A = __builtin_bit_cast(wsoft_u8, a), B = __builtin_bit_cast(wsoft_u8, b);
    float d0 = c[0], d1 = c[1], d2 = c[2], d3 = c[3], d4 = c[4], d5 = c[5], d6 = c[6], d7 = c[7];
    __asm volatile("s_nop 1\n"
    DOT2DPP " %0, %8, %16 row_share:0 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %1, %8, %16 row_share:2 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %2, %8, %16 row_share:4 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %3, %8, %16 row_share:6 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %4, %8, %16 row_share:8 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %5, %8, %16 row_share:10 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %6, %8, %16 row_share:12 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %7, %8, %16 row_share:14 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %0, %9, %17 row_share:0 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %1, %9, %17 row_share:2 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %2, %9, %17 row_share:4 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %3, %9, %17 row_share:6 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %4, %9, %17 row_share:8 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %5, %9, %17 row_share:10 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %6, %9, %17 row_share:12 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %7, %9, %17 row_share:14 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %0, %10, %18 row_share:0 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %1, %10, %18 row_share:2 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %2, %10, %18 row_share:4 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %3, %10, %18 row_share:6 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %4, %10, %18 row_share:8 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %5, %10, %18 row_share:10 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %6, %10, %18 row_share:12 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %7, %10, %18 row_share:14 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %0, %11, %19 row_share:0 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %1, %11, %19 row_share:2 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %2, %11, %19 row_share:4 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %3, %11, %19 row_share:6 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %4, %11, %19 row_share:8 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %5, %11, %19 row_share:10 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %6, %11, %19 row_share:12 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %7, %11, %19 row_share:14 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %0, %12, %20 row_share:0 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %1, %12, %20 row_share:2 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %2, %12, %20 row_share:4 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %3, %12, %20 row_share:6 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %4, %12, %20 row_share:8 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %5, %12, %20 row_share:10 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %6, %12, %20 row_share:12 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %7, %12, %20 row_share:14 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %0, %13, %21 row_share:0 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %1, %13, %21 row_share:2 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %2, %13, %21 row_share:4 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %3, %13, %21 row_share:6 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %4, %13, %21 row_share:8 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %5, %13, %21 row_share:10 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %6, %13, %21 row_share:12 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %7, %13, %21 row_share:14 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %0, %14, %22 row_share:0 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %1, %14, %22 row_share:2 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %2, %14, %22 row_share:4 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %3, %14, %22 row_share:6 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %4, %14, %22 row_share:8 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %5, %14, %22 row_share:10 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %6, %14, %22 row_share:12 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %7, %14, %22 row_share:14 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %0, %15, %23 row_share:0 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %1, %15, %23 row_share:2 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %2, %15, %23 row_share:4 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %3, %15, %23 row_share:6 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %4, %15, %23 row_share:8 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %5, %15, %23 row_share:10 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %6, %15, %23 row_share:12 row_mask:0xf bank_mask:0xf\n"
    DOT2DPP " %7, %15, %23 row_share:14 row_mask:0xf bank_mask:0xf\n"
          : "+&v"(d0), "+&v"(d1), "+&v"(d2), "+&v"(d3), "+&v"(d4), "+&v"(d5), "+&v"(d6), "+&v"(d7)
          : "v"(A[0]), "v"(A[1]), "v"(A[2]), "v"(A[3]), "v"(A[4]), "v"(A[5]), "v"(A[6]), "v"(A[7]),
            "v"(B[0]), "v"(B[1]), "v"(B[2]), "v"(B[3]), "v"(B[4]), "v"(B[5]), "v"(B[6]), "v"(B[7]));
    return (wsoft_f8)(d0, d1, d2, d3, d4, d5, d6, d7);
}
// the row of a 16-row A operand that lane l loads for wmma_soft
static inline int wsoft_row(int l) { return (l + (l >> 4)) & 15; }

#endif
