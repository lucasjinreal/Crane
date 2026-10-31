// SPDX-License-Identifier: MIT
/**
 * Stream-K fixup for the MMA flash-attention kernels. When a launch has
 * fewer blocks than (output tile, KV block) units, a block's seam can fall
 * inside a tile. Every block of such a tile except the first writes an
 * unnormalized partial plus its softmax max and row sum to the F32 fixup
 * buffer, and `crane_fattn_stream_k_fixup` merges them into `dst`.
 *
 * Adapted from llama.cpp's `flash_attn_stream_k_fixup_general`
 * (`fattn_common.cuh`). `dst` holds the 16-bit output, so it cannot carry
 * the partial of the block that finishes a tile the way llama.cpp's F32
 * `dst` does. That partial goes to a second data region of the fixup
 * buffer instead, and the merged value is rounded once to `dst`'s type.
 *
 * Fixup buffer layout in `float2`, for a grid of `G` blocks:
 *
 *   [G*ncols needs_fixup meta | G*ncols is_fixup meta |
 *    G*ncols*(DV/2) is_fixup data | G*ncols*(DV/2) needs_fixup data]
 *
 * Meta is `(KQ max, KQ row sum)` per column. A block's `needs_fixup` segment
 * (the end of a tile it did not start) uses meta slot `blockIdx.x` and data
 * slot `G + blockIdx.x`. Its `is_fixup` segment (the start of a tile it does
 * not finish) uses meta slot `G + blockIdx.x` and data slot `blockIdx.x`.
 */
#pragma once

#include "fattn_common.cuh"

// Writes a partial that drops out of the merge: the kernels' initial KQ max,
// a zero row sum and a zero accumulator. For a segment whose KV range lies
// wholly past the causal bound, which never runs the tile loop.
template <int DV, int ncols>
static __device__ __forceinline__ void crane_fattn_write_empty_partial(
        float2 * __restrict__ dst_fixup, const int meta_slot, const int data_slot) {
    const int tid      = threadIdx.y*blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x*blockDim.y;

    float2 * meta = dst_fixup + meta_slot*ncols;
    for (int jc = tid; jc < ncols; jc += nthreads) {
        meta[jc] = make_float2(-FLT_MAX/2.0f, 0.0f);
    }
    float2 * data = dst_fixup + gridDim.x*(2*ncols) + data_slot*(ncols*(DV/2));
    for (int i = tid; i < ncols*(DV/2); i += nthreads) {
        data[i] = make_float2(0.0f, 0.0f);
    }
}

static __device__ __forceinline__ void crane_fattn_store(half * dst, const float val) {
    *dst = __float2half_rn(val);
}

static __device__ __forceinline__ void crane_fattn_store(nv_bfloat16 * dst, const float val) {
    *dst = __float2bfloat16_rn(val);
}

// One block per (main-kernel block, Q row in the tile, GQA head in the
// tile), one thread per output element of the row. Only blocks that
// finished a tile they did not start do any work. They walk back over the
// earlier blocks of that tile and merge their partials in a fixed order, so
// the result is deterministic.
template <int DV, int ncols1, int ncols2, typename T_dst>
static __device__ __forceinline__ void crane_fattn_stream_k_fixup(
        T_dst        * __restrict__ dst,
        const float2 * __restrict__ dst_fixup,
        const int ne01, const int ne02,
        const int gqa_ratio,
        const int total_work,
        const uint3 fd_iter_k_j_z_ne12,
        const uint3 fd_iter_k_j_z,
        const uint3 fd_iter_k_j,
        const uint3 fd_iter_k) {
    constexpr int ncols = ncols1*ncols2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*ncols2 + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data       = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);
    const float * dst_fixup_needs_data = dst_fixup_data + gridDim.x*(ncols*DV);

    const int kbc0      = int64_t(bidx0 + 0)*total_work / gridDim.x;
    const int kbc0_stop = int64_t(bidx0 + 1)*total_work / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, fd_iter_k) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, fd_iter_k) == fastdiv(kbc0_stop, fd_iter_k) && fastmodulo(kbc0_stop, fd_iter_k) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(kbc0, fd_iter_k_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y, fd_iter_k_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y, fd_iter_k_j);
    const uint2 dm3 = fast_div_modulo(dm2.y, fd_iter_k);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm3.x;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*DV + jt*ne02*(ncols1*DV) + zt_Q*DV + (j*ne02 + c)*DV + tid;

    // The partial of the block that finished the tile.
    float dst_val = dst_fixup_needs_data[bidx0*ncols*DV + jc*DV + tid];
    float max_val;
    float rowsum;
    {
        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    const int tile_kbc0 = fastdiv(kbc0, fd_iter_k);
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = int64_t(bidx)*total_work / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*DV + jc*DV + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, fd_iter_k) == 0 || fastdiv(kbc, fd_iter_k) < tile_kbc0) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    crane_fattn_store(dst, dst_val / rowsum);
}

// The `extern "C"` fixup entry for one kernel instantiation. Expanded by
// `DECL_FATTN_MMA_F16_CASE`/`DECL_FATTN_MMA_BF16_CASE`, so every MMA kernel
// has a fixup with the same `(DV, ncols1, ncols2)`.
#define DECL_CRANE_FATTN_FIXUP(tag_, T_dst_, DV_, ncols1_, ncols2_)                                     \
    extern "C" __global__                                                                              \
    __launch_bounds__(DV_, 1)                                                                          \
    void crane_fattn_fixup_##tag_##_d##DV_##_c##ncols1_##_s##ncols2_(                                  \
        T_dst_ * dst, const float2 * dst_fixup,                                                        \
        int32_t ne01, int32_t ne02, int32_t gqa_ratio, int32_t total_work,                              \
        uint3 fd_iter_k_j_z_ne12, uint3 fd_iter_k_j_z, uint3 fd_iter_k_j, uint3 fd_iter_k)              \
    {                                                                                                    \
        crane_fattn_stream_k_fixup<DV_, ncols1_, ncols2_>(                                             \
            dst, dst_fixup, ne01, ne02, gqa_ratio, total_work,                                          \
            fd_iter_k_j_z_ne12, fd_iter_k_j_z, fd_iter_k_j, fd_iter_k);                                 \
    }
