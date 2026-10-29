// MIT License
//
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Derived from llama.cpp's ggml/src/ggml-cuda/fattn-mma-f16.cuh: no upstream
// bf16 MMA flash-attention kernel exists to vendor (upstream's bf16 support
// is confined to the scalar vec/decode kernel), so this is new code written
// against this project's own fattn_mma_f16.cuh, not a direct port.
// SPDX-License-Identifier: MIT
/**
 * Tensor-core (MMA) flash-attention prefill kernel for native BF16 K/V,
 * avoiding the BF16->F16 downcast `gpu_flash_attn.rs` otherwise applies
 * before every kernel call (F16's 5 exponent bits overflow to `inf` for
 * BF16-range activation values F16 can't represent; BF16 keeps F32's 8
 * exponent bits).
 *
 * Differences from `fattn_mma_f16.cuh`, all deliberate initial-landing
 * simplifications rather than oversights:
 *
 * - **No native bf16 MMA on Turing or Volta.** NVIDIA's bf16 tensor-core
 *   instruction requires Ampere (`AMPERE_MMA_AVAILABLE`); there is no
 *   pre-Ampere fallback the way f16 has one down to Turing/Volta. The
 *   active kernel body is gated on `AMPERE_MMA_AVAILABLE ||
 *   AMD_WMMA_AVAILABLE || AMD_MFMA_AVAILABLE`.
 * - **No `cols_per_warp == 8` (`ncols1*ncols2 == 8`) tile shape.** The f16
 *   kernel's narrow `ncols == 8` specialization exists to match a
 *   same-precision (`half2`) PV accumulator path; bf16's PV accumulator is
 *   `float` on every architecture (bf16 MMA has no same-precision
 *   accumulate), so this kernel always uses the wide (16-column) tile
 *   shape instead, at the cost of a little occupancy for `ncols1*ncols2 ==
 *   8` launches. `fattn_mma_hd*_bf16.cu` instantiates only `ncols1*ncols2`
 *   in {16, 32, 64}, and `ops::fused_ops::fattn::select_mma_ncols_bf16`
 *   never selects a combination resolving to 8.
 * - **PV accumulator type.** Because bf16 MMA only ever produces `float`
 *   (no `half2`-equivalent same-precision accumulate on any target
 *   architecture), `T_C_VKQ` here is always `float` and always the same
 *   tile shape/type as `T_C_KQ` (the QK^T accumulator, already `float` in
 *   the f16 kernel too) - not a smaller/packed type the way f16's `T_C_VKQ`
 *   is. The softmax-rescale and VKQ-combine code paths that assume a
 *   `half2` PV accumulator in `fattn_mma_f16.cuh` are replaced with a
 *   `float`-only path; the index arithmetic matches `fattn_mma_f16.cuh`'s
 *   own existing `T_C_KQ`-based column derivation (`(l/2) % 2` on
 *   NVIDIA/`cols_per_thread == 2`, `0` on AMD/`cols_per_thread == 1`),
 *   confirmed against that code since `T_C_VKQ` here has the exact same
 *   physical layout as `T_C_KQ`.
 * - **No shared-memory bank-conflict swizzle (NVIDIA-only perf path) and no
 *   multi-stage (`cp.async`) pipelining.** Both are pure performance
 *   optimizations layered on top of the same correctness-bearing tile
 *   layout; deferred so this initial bf16 landing only has one shared-
 *   memory addressing scheme and one K/V loading path to verify. `nstages`
 *   is hardcoded to 0 and `swz_K`/`swz_V` to `false` throughout (matching
 *   AMD's existing behavior in the f16 kernel, where swizzle is already
 *   always disabled).
 * - The config tables, `get_cols_per_thread`, and the mask-tile loader
 *   (`flash_attn_ext_bf16_load_mask`) are copied unchanged from
 *   `fattn_mma_f16.cuh` (separate translation unit, so duplicate symbol
 *   names are not an ODR conflict): the mask is always F16 regardless of
 *   K/V dtype, and the tuning tables are reused as an explicit untuned
 *   starting point (see `ops::fused_ops::fattn`'s module doc).
 *
 * Usage (from an instantiation file):
 * ```cpp
 * #include "fattn_mma_bf16.cuh"
 * DECL_FATTN_MMA_BF16_CASE(128, 128, 16, 1);
 * ```
 */
#include "crane_fattn_shim.cuh"
#include "mma.cuh"
#include "fattn_common.cuh"

using namespace ggml_cuda_mma;

// Config options for the MMA kernel. Identical shape to fattn_mma_f16.cuh's
// fattn_mma_config; duplicated (not shared) because this is a separate
// translation unit and the two kernel families are never linked together.
struct fattn_mma_config {
    int  nthreads;
    int  occupancy;
    int  nbatch_fa;
    int  nbatch_K2;
    int  nbatch_V2;
    int  nbatch_combine;
    int  nstages_target;
    bool Q_in_reg;

    constexpr __host__ __device__ fattn_mma_config(
            int nthreads, int occupancy, int nbatch_fa, int nbatch_K2, int nbatch_V2, int nbatch_combine, int nstages_target, bool Q_in_reg) :
        nthreads(nthreads), occupancy(occupancy), nbatch_fa(nbatch_fa), nbatch_K2(nbatch_K2), nbatch_V2(nbatch_V2), nbatch_combine(nbatch_combine),
        nstages_target(nstages_target), Q_in_reg(Q_in_reg) {}
};

#define GGML_CUDA_FATTN_MMA_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads_, occupancy_, nbatch_fa_, nbatch_K2_, nbatch_V2_, nbatch_combine_, nstages_target_, Q_in_reg_) \
    if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                                                                                       \
        static_assert((nthreads_)       % 32 == 0 && (nthreads_)       <= 512, "bad nthreads");                                                                    \
        static_assert(                               (occupancy_)      <=   8, "bad occupancy");                                                                   \
        static_assert((nbatch_fa_)      % 32 == 0 && (nbatch_fa_)      <= 256, "bad nbatch_fa");                                                                   \
        static_assert((nbatch_K2_)      %  4 == 0 && (nbatch_K2_)      <= 512, "bad nbatch_K2");                                                                   \
        static_assert((nbatch_V2_)      %  4 == 0 && (nbatch_V2_)      <= 256, "bad nbatch_V2");                                                                   \
        static_assert((nbatch_combine_) %  4 == 0 && (nbatch_combine_) <= 128, "bad nbatch_combine");                                                              \
        static_assert((nstages_target_)      >= 1 && (nstages_target_) <=   2, "bad nstages_target");                                                              \
        return fattn_mma_config{(nthreads_), (occupancy_), (nbatch_fa_), (nbatch_K2_), (nbatch_V2_), (nbatch_combine_), (nstages_target_), (Q_in_reg_)};           \
    }                                                                                                                                                              \

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_ampere(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 128, 2,  64,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 128, 2,  64,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 128, 2,  64,  32,  32,  32, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 128, 2,  64,  64,  64,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16,  64, 4,  32, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  32, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  32, 128, 128, 128, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32, 256, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128, 128, 1, false);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_rdna(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 128, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 128, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 128, 2,  64,  32,  32,  32, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16,  64, 2,  32,  64,  64,  64, 1, true);
    // nbatch_fa halved vs fattn_mma_f16.cuh's own RDNA table at this
    // (DKQ, ncols): bf16's PV accumulator is always float (no half2-packed
    // accumulate exists for bf16 WMMA, see this file's module doc), twice
    // the register footprint of F16's packed accumulator at this shape.
    // Trading tile size for occupancy here measurably helped
    // ops::fused_ops::fattn_mma::gpu_tests::bf16_vs_f16_causal_timing_hd128_gqa4.
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 128, 2,  32,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 128, 2,  32,  64,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16,  64, 2,  32, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 256, 2,  64, 128, 128,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 256, 2,  64, 128, 128,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 128, 2,  32, 128, 128, 128, 1, true);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static constexpr __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int DKQ, const int DV, const int ncols) {
#if defined(AMPERE_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_rdna(DKQ, DV, ncols);
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols);
    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
#endif // defined(AMPERE_MMA_AVAILABLE)
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nthreads(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nthreads;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_occupancy(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).occupancy;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_fa(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_fa;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_K2(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_K2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_V2(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_V2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_combine(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_combine;
}

static constexpr __device__ bool ggml_cuda_fattn_mma_get_Q_in_reg(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).Q_in_reg;
}

static constexpr __device__ int get_cols_per_thread() {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    return 1; // AMD has a single column per thread.
#else
    return 2; // This is specifically KQ columns.
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

// bf16 counterpart of fattn_mma_f16.cuh's flash_attn_ext_f16_load_tile:
// identical byte-level copy logic (K/V data is never arithmetically
// touched while loading), just nv_bfloat162 instead of half2. Unlike the
// f16 version, the `swz` (shared-memory swizzle) and `use_cp_async`
// (multi-stage pipelining) branches are removed outright rather than kept
// as `if constexpr (false)` dead code: this kernel always instantiates
// both `false` (see this file's module doc), and the removed branches
// reference `ggml_cuda_fattn_smem_swizzle::`/`cp_async_cg_16` from
// `swizzle.cuh`/`cp_async.cuh`, which this file does not include. Those are
// non-dependent names inside the discarded branch, so two-phase lookup
// requires them to be visible at template-definition time regardless of
// `if constexpr` — confirmed by a real "use of undeclared identifier"
// build failure keeping them in without the corresponding `#include`.
// `swz`/`use_cp_async` stay as (unused) template parameters so call sites
// (which already always pass `false`) don't need to change.
template<int stride_tile, bool swz, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check, bool use_sparse>
static __device__ __forceinline__ void flash_attn_ext_bf16_load_tile(
        const nv_bfloat162 * const __restrict__ KV, nv_bfloat162 * const __restrict__ tile_KV, const int D2, const int stride_KV,
        const int k_VKQ_0, const int i_sup, const int32_t * const __restrict__ indices) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int h2_per_chunk = 16/sizeof(nv_bfloat162);
    const int chunks_per_row = D2 / h2_per_chunk;
    const nv_bfloat162 zero[4] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
    auto load = [&] __device__ (const int n) {
        const int stride_k = 32 >> n;
        const int k0_start = stride_k == 32 ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
        const int k0_stop  =                      chunks_per_row - chunks_per_row % (1*stride_k);
        const int stride_i = warp_size / stride_k;

        if (k0_start == k0_stop) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
            const int i = i0 + threadIdx.y*stride_i + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

            if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                break;
            }

#pragma unroll
            for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                const nv_bfloat162 * src;
                if constexpr (use_sparse) {
                    const int32_t index = i < i_sup ? indices[k_VKQ_0 + i] : -1;
                    src = index >= 0 ? KV + int64_t(index)*stride_KV + k*h2_per_chunk : zero;
                } else {
                    src = !oob_check || i < i_sup ? KV + int64_t(k_VKQ_0 + i)*stride_KV + k*h2_per_chunk : zero;
                }
                ggml_cuda_memcpy_1<16>(tile_KV + i*stride_tile + k*4, src);
            }
        }
    };
    ggml_cuda_unroll<6>{}(load);
}

// Mask tile loader: byte-identical to fattn_mma_f16.cuh's
// flash_attn_ext_f16_load_mask (the mask is always F16, independent of K/V
// dtype), duplicated rather than shared since this is a separate
// translation unit. Unlike the f16 version, the `use_cp_async` branch is
// removed outright rather than kept as dead code — see
// flash_attn_ext_bf16_load_tile's doc comment for why. `use_cp_async` stays
// an (unused) template parameter so call sites (which already always pass
// `false`) don't need to change.
template<int ncols1, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check, bool use_sparse>
static __device__ __forceinline__ void flash_attn_ext_bf16_load_mask(
        const half * const __restrict__ mask_h, half * const __restrict__ tile_mask,
        const int stride_mask, const int k_VKQ_0, const int i_sup, const int j0, const uint3 ne01,
        const int32_t * const __restrict__ indices) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    if constexpr (oob_check || use_sparse) {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                if constexpr (use_sparse) {
                    const int32_t index = i < i_sup ? indices[k_VKQ_0 + i] : -1;
                    tile_mask[j_sram*(nbatch_fa + 8) + i] = index >= 0 ? mask_h[int64_t(j_vram)*stride_mask + index] : half(-INFINITY);
                } else {
                    tile_mask[j_sram*(nbatch_fa + 8) + i] = i < i_sup ? mask_h[int64_t(j_vram)*stride_mask + k_VKQ_0 + i] : half(0.0f);
                }
            }
        }
    } else if constexpr (nbatch_fa < 2*warp_size) {
        constexpr int cols_per_warp = 2*warp_size/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y*cols_per_warp + threadIdx.x / (warp_size/cols_per_warp);
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + stride_j > ncols1 && j_sram >= ncols1) {
                break;
            }

            const int i = threadIdx.x % (warp_size/cols_per_warp);

            ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + 2*i, mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + 2*i);
        }
    } else {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += 2*warp_size) {
                const int i = i0 + 2*threadIdx.x;

                ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + i, mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + i);
            }
        }
    }
}

// Always the wide (cols_per_warp == 16) tile shape: see this file's module
// doc for why the f16 kernel's narrow (ncols == 8) specialization and its
// half2 PV accumulator have no bf16 equivalent here. `ncols` stays a
// template parameter (unused in the body) only so call sites matching
// fattn_mma_f16.cuh's `mma_tile_sizes<DV, ncols>::T_*` shape keep working
// unchanged.
#if defined(AMPERE_MMA_AVAILABLE)
template<int DV, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, nv_bfloat162>; // row-major
    using T_B_KQ  = tile<16,  8, nv_bfloat162>; // column-major
    using T_C_KQ  = tile<16, 16, float>;        // column-major
    using T_A_VKQ = tile<16,  8, nv_bfloat162>; // row-major
    using T_B_VKQ = tile<16,  8, nv_bfloat162>; // column-major
    using T_C_VKQ = tile<16, 16, float>;        // column-major
};
#elif defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
template<int DV, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, nv_bfloat162, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<16,  8, nv_bfloat162, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_KQ  = tile<16, 16, float,        DATA_LAYOUT_I_MAJOR>;         // column-major
    using T_A_VKQ = tile<32,  8, nv_bfloat162, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_VKQ = tile<16,  8, nv_bfloat162, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_VKQ = tile<16, 16, float,        DATA_LAYOUT_I_MAJOR>;         // column-major
};
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
template<int DV, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, nv_bfloat162>; // row-major
    using T_B_KQ  = tile<16,  8, nv_bfloat162>; // column-major
    using T_C_KQ  = tile<16, 16, float>;        // column-major
    using T_A_VKQ = tile<16,  8, nv_bfloat162>; // row-major
    using T_B_VKQ = tile<16,  8, nv_bfloat162>; // column-major
    using T_C_VKQ = tile<16, 16, float>;        // column-major
};
#else // Turing-without-Ampere, or Volta: no native bf16 MMA, kernel body is NO_DEVICE_CODE.
template<int DV, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, nv_bfloat162>;
    using T_B_KQ  = tile<16,  8, nv_bfloat162>;
    using T_C_KQ  = tile<16, 16, float>;
    using T_A_VKQ = tile<16,  8, nv_bfloat162>;
    using T_B_VKQ = tile<16,  8, nv_bfloat162>;
    using T_C_VKQ = tile<16, 16, float>;
};
#endif // defined(AMPERE_MMA_AVAILABLE)

// bf16 counterpart of fattn_mma_f16.cuh's flash_attn_ext_f16_iter, with the
// narrow (cols_per_warp == 8) branches removed (see this file's module doc)
// and the half2-only softmax-rescale/B-tile-conversion code replaced by a
// float path. This file never instantiates nstages > 1, so the K-tile
// preload-for-next-iteration block fattn_mma_f16.cuh has under `if
// constexpr (nstages > 1)` is dropped entirely rather than kept as dead
// code.
template<int DKQ, int DV, int ncols1, int ncols2, int nwarps,
    bool use_logit_softcap, bool V_is_K_view, bool use_sparse, bool needs_fixup, bool is_fixup, bool last_iter, bool oob_check,
    typename T_A_KQ, typename T_B_KQ, typename T_C_KQ, typename T_A_VKQ, typename T_B_VKQ, typename T_C_VKQ>
static __device__ __forceinline__ void flash_attn_ext_bf16_iter(
        const float2 * const __restrict__ Q_f2,
        const nv_bfloat162 * const __restrict__ K_h2,
        const nv_bfloat162 * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        const int32_t * const __restrict__ indices,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        nv_bfloat162 * const __restrict__ tile_Q,
        nv_bfloat162 * const __restrict__ tile_K,
        nv_bfloat162 * const __restrict__ tile_V,
        half         * const __restrict__ tile_mask,
        T_B_KQ       * const __restrict__ Q_B,
        T_C_VKQ      * const __restrict__ VKQ_C,
        float        * const __restrict__ KQ_max,
        float        * const __restrict__ KQ_rowsum,
        const int jt,
        const int kb0,
        const int k_VKQ_sup) {
#if defined(AMPERE_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    constexpr int  warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int  ncols           = ncols1 * ncols2;
    constexpr int  cols_per_warp   = T_B_KQ::I; // Always 16: see this file's module doc.
    constexpr int  cols_per_thread = get_cols_per_thread();
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // Number of parallel CUDA warps per Q column.
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, ncols);
    constexpr int  nbatch_V2       = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, ncols);
    constexpr bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg (DKQ, DV, ncols);

    // Swizzle disabled (see this file's module doc): the stride is always
    // the "disabled" `n + 4` formula fattn_mma_f16.cuh's
    // ggml_cuda_fattn_smem_swizzle::tile_stride falls back to on AMD.
    constexpr int stride_tile_K = nbatch_K2 + 4;
    constexpr int stride_tile_V = V_is_K_view ? stride_tile_K : nbatch_V2 + 4;

    const int k_VKQ_0 = kb0 * nbatch_fa;
    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];

    // nstages is always 0 (see this file's module doc), so the mask is
    // always loaded synchronously here; there is no K-tile preload for the
    // next iteration to do multi-stage.
    if (ncols2 > 1 || mask_h) {
        flash_attn_ext_bf16_load_mask<ncols1, nwarps, nbatch_fa, false, oob_check, use_sparse>
            (mask_h, tile_mask, stride_mask, k_VKQ_0, k_VKQ_sup, jt*ncols1, ne01, indices);
    }

    // For MLA K and V have the same data.
    // Therefore, iterate over K in reverse and later re-use the data if possible.
#pragma unroll
    for (int k0_start = (DKQ/2-1) - (DKQ/2-1) % nbatch_K2; k0_start >= 0; k0_start -= nbatch_K2) {
        const int k0_stop = k0_start + nbatch_K2 < DKQ/2 ? k0_start + nbatch_K2 : DKQ/2;
        const int k0_diff = k0_stop - k0_start;

        flash_attn_ext_bf16_load_tile<stride_tile_K, false, nwarps, nbatch_fa, false, oob_check, use_sparse>
            (K_h2 + k0_start, tile_K, k0_diff, stride_K, k_VKQ_0, k_VKQ_sup, indices);
        __syncthreads();

        // Calculate tile of KQ (always the wide, column-major path: see
        // this file's module doc for why the narrow cols_per_warp == 8
        // path has no bf16 counterpart):
        if constexpr (Q_in_reg) {
#pragma unroll
            for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;
#pragma unroll
                for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                    T_A_KQ K_A;
                    load_ldmatrix(K_A, tile_K + i_KQ_0*stride_tile_K + (k_KQ_0 - k0_start), stride_tile_K);
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    // AMD matrix C is column-major.
                    mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
#else
                    // swap A and B for CUDA.
                    mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[k_KQ_0/T_A_KQ::J], K_A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                }
            }
        } else {
            constexpr int stride_tile_Q = DKQ/2 + 4;
#pragma unroll
            for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                load_ldmatrix(Q_B[0], tile_Q + (threadIdx.y / np)*(T_B_KQ::I*stride_tile_Q) + k_KQ_0, stride_tile_Q);

#pragma unroll
                for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                    const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;

                    T_A_KQ K_A;
                    load_ldmatrix(K_A, tile_K + i_KQ_0*stride_tile_K + (k_KQ_0 - k0_start), stride_tile_K);

#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    // AMD matrix C is column-major.
                    mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[0]);
#else
                    // swap A and B for CUDA.
                    mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[0], K_A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                }
            }
        }

        __syncthreads(); // Only needed if tile_K == tile_V.
    }

    if (use_logit_softcap) {
#pragma unroll
        for (int i = 0; i < nbatch_fa/(np*T_C_KQ::J); ++i) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                KQ_C[i].x[l] = logit_softcap*tanhf(KQ_C[i].x[l]);
            }
        }
    }

    float KQ_max_new[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max_new[col] = KQ_max[col];
    }
    float KQ_rowsum_add[cols_per_thread] = {0.0f};

    if (ncols2 > 1 || mask_h) {
#pragma unroll
        for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::J) {
            const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::J;

            // The mask is stored as 16 bit half values, loading them as 32 bit half2 values is preferred in terms of speed.
            // However, this is not possible for RDNA3 where 2 consecutive l indices are not consecutive in the mask memory layout.
#ifdef RDNA3
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                const int i = i0 + T_C_KQ::get_j(l);
                const int j = ((threadIdx.y / np)*cols_per_warp + T_C_KQ::get_i(l)) / ncols2;

                KQ_C[i00/(np*T_C_KQ::J)].x[l] += __half2float(tile_mask[j*(nbatch_fa + 8) + i]);
            }
#else
#pragma unroll
            for (int l0 = 0; l0 < T_C_KQ::ne; l0 += 2) {
                const int i = (i0 + T_C_KQ::get_j(l0)) / 2;
                const int j = ((threadIdx.y / np)*cols_per_warp + T_C_KQ::get_i(l0)) / ncols2;

                const float2 tmp = __half22float2(((const half2 *) tile_mask)[j*(nbatch_fa/2 + 4) + i]);
                KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 0] += slope*tmp.x;
                KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 1] += slope*tmp.y;
            }
#endif // RDNA3
        }
    }

    // Calculate softmax for each KQ column using the current max. value.
    // The divisor is stored in KQ_rowsum and will be applied at the end.
#pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                constexpr int KQ_idx = 0;
#else
                const int KQ_idx = (l/2) % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[k0/(np*T_C_KQ::J)].x[l] + FATTN_KQ_MAX_OFFSET);
            }
        }
    }

#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
#if defined(AMD_MFMA_AVAILABLE)
        constexpr int offset_first = 32;
        constexpr int offset_last  = 16;
#elif defined(AMD_WMMA_AVAILABLE)
        constexpr int offset_first = 16;
        constexpr int offset_last  = 16;
#else // Ampere
        constexpr int offset_first = 2;
        constexpr int offset_last  = 1;
#endif // defined(AMD_MFMA_AVAILABLE)
#pragma unroll
        for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
            KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, warp_size));
        }
    }

#pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                constexpr int KQ_idx = 0;
#else
                const int KQ_idx = (l/2) % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                KQ_C[k0/(np*T_C_KQ::J)].x[l] = expf(KQ_C[k0/(np*T_C_KQ::J)].x[l] - KQ_max_new[KQ_idx]);
                KQ_rowsum_add[KQ_idx] += KQ_C[k0/(np*T_C_KQ::J)].x[l];
            } else {
                KQ_C[k0/(np*T_C_KQ::J)].x[l] = 0.0f;
            }
        }
    }

    {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const float KQ_max_diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new[col];

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
        }

        // T_C_VKQ is always float here (see this file's module doc), with
        // the exact same physical layout as T_C_KQ: reuse T_C_KQ's own
        // column-index derivation above (KQ_idx) rather than f16's
        // half2-packed scheme.
#pragma unroll
        for (int i = 0; i < DV/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                constexpr int KQ_idx = 0;
#else
                const int KQ_idx = (l/2) % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                VKQ_C[i].x[l] *= KQ_max_scale[KQ_idx];
            }
        }
    }

    // Convert KQ C tiles into B tiles for VKQ calculation (always the
    // non-transposed path: see this file's module doc for why the
    // cols_per_warp == 8/get_transposed path has no bf16 counterpart).
    T_B_VKQ B[nbatch_fa/(np*2*T_B_VKQ::J)];
    static_assert(nbatch_fa % (np*2*T_B_VKQ::J) == 0, "bad loop size");
    for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
        B[k] = get_bf16(KQ_C[k]);
    }

    // Calculate VKQ tile, need to use logical rather than physical elements for i0 due to transposition of V:
#pragma unroll
    for (int i0_start = 0; i0_start < DV; i0_start += 2*nbatch_V2) {
        static_assert(DV % (2*nbatch_V2) == 0, "bad loop size");
        const int i0_stop = i0_start + 2*nbatch_V2;

        const int i0_diff = i0_stop - i0_start;
        if (!V_is_K_view || i0_stop > 2*nbatch_K2) {
            constexpr bool use_cp_async = false;
            flash_attn_ext_bf16_load_tile<stride_tile_V, false, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                (V_h2 + i0_start/2, tile_V, i0_diff/2, stride_V, kb0*nbatch_fa, k_VKQ_sup, indices);
            __syncthreads();
        }
        const nv_bfloat162 * tile_V_i = !V_is_K_view || i0_stop > 2*nbatch_K2 ? tile_V : tile_V + i0_start/2;

#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += T_A_VKQ::I) {
            static_assert((nbatch_fa/2) % (np*T_A_VKQ::J) == 0, "bad loop size");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa/2; k00 += np*T_A_VKQ::J) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::J;

                T_A_VKQ A; // Transposed in SRAM but not in registers, gets transposed on load.
                load_ldmatrix_trans(A, tile_V + ((int)(tile_V_i - tile_V) + 2*k0*stride_tile_V + (i_VKQ_0 - i0_start)/2), stride_tile_V);
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                // AMD matrix C is column-major.
                mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], A, B[k00/(np*T_A_VKQ::J)]);
#else
                // swap A and B for CUDA.
                mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], B[k00/(np*T_A_VKQ::J)], A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
            }
        }

        __syncthreads(); // Only needed if tile_K == tile_V.
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02,
        stride_K, stride_V, stride_mask,
        tile_Q, tile_K, tile_V, tile_mask,
        Q_B, VKQ_C, KQ_max, KQ_rowsum, kb0);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

// bf16 counterpart of fattn_mma_f16.cuh's flash_attn_ext_f16_process_tile.
// Same simplifications as flash_attn_ext_bf16_iter above (no cols_per_warp
// == 8, no swizzle, no multi-stage pipelining); additionally, the VKQ
// combine buffer is written/read through `nv_bfloat16`/`nv_bfloat162`
// rather than `half`/`half2`: fattn_mma_f16.cuh's float-T_C_VKQ combine
// path (never exercised by any f16 instantiation at Crane's head dims,
// since f16's float-T_C_VKQ case only occurs for DV in {80, 112}, which
// Crane never instantiates) narrows through `half` before combining, which
// would reintroduce this kernel's entire reason for existing (bf16's wider
// range) at the combine step. This kernel's VKQ accumulator is float on
// every architecture, so this path is not a rarely-hit edge case here: it
// is the only path, and keeping it in `nv_bfloat16`/F32 avoids that
// narrowing.
template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, bool use_logit_softcap, bool V_is_K_view, bool use_sparse, bool needs_fixup, bool is_fixup>
static __device__ __forceinline__ void flash_attn_ext_bf16_process_tile(
        const float2 * const __restrict__ Q_f2,
        const nv_bfloat162 * const __restrict__ K_h2,
        const nv_bfloat162 * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        const int32_t * const __restrict__ indices,
        const float  * const __restrict__ sinks_f,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int gqa_ratio,
        const int ne11,
        const int stride_Q1,
        const int stride_Q2,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        const int jt,
        const int zt_gqa,
        const int kb0_start,
        const int kb0_stop) {
#if defined(AMPERE_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int ncols = ncols1 * ncols2;
    using     T_A_KQ    = typename mma_tile_sizes<DV, ncols>::T_A_KQ;
    using     T_B_KQ    = typename mma_tile_sizes<DV, ncols>::T_B_KQ;
    using     T_C_KQ    = typename mma_tile_sizes<DV, ncols>::T_C_KQ;
    using     T_A_VKQ   = typename mma_tile_sizes<DV, ncols>::T_A_VKQ;
    using     T_B_VKQ   = typename mma_tile_sizes<DV, ncols>::T_B_VKQ;
    using     T_C_VKQ   = typename mma_tile_sizes<DV, ncols>::T_C_VKQ;

    constexpr int  cols_per_warp   = T_B_KQ::I; // Always 16: see this file's module doc.
    constexpr int  cols_per_thread = get_cols_per_thread();
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // Number of parallel CUDA warps per Q column.
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols);
    constexpr int  nbatch_V2       = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols);
    constexpr int  nbatch_combine  = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols);
    constexpr bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols);

    if (cols_per_warp > ncols) {
        NO_DEVICE_CODE;
        return;
    }

    static_assert(nwarps * (cols_per_warp/ncols2) % ncols1 == 0, "bad nwarps");

    constexpr int stride_tile_Q = DKQ/2 + 4;
    // Swizzle disabled (see this file's module doc): always the "disabled"
    // `n + 4` stride formula.
    constexpr int stride_tile_K = nbatch_K2 + 4;
    constexpr int stride_tile_V = V_is_K_view ? stride_tile_K : nbatch_V2 + 4;
    constexpr int stride_tile_KV_max = stride_tile_K > stride_tile_V ? stride_tile_K : stride_tile_V;

    extern __shared__ nv_bfloat162 tile_Q[];
    nv_bfloat162 * tile_K    = Q_in_reg ? tile_Q : tile_Q + ncols * stride_tile_Q;
    nv_bfloat162 * tile_V    = tile_K;
    half         * tile_mask = (half *) (tile_V + nbatch_fa * stride_tile_KV_max);

    T_B_KQ    Q_B[(Q_in_reg ? DKQ/(2*T_B_KQ::J) : 1)];
    // T_C_VKQ is always float (see this file's module doc), so unlike
    // fattn_mma_f16.cuh there is no per-architecture tile-shape ternary
    // here: DV/T_C_VKQ::J is the right physical size for every target.
    T_C_VKQ VKQ_C[DV/T_C_VKQ::J];

    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    // Load Q data into tile_Q, either temporarily or permanently.
    // Q in registers is faster, but register pressure is the biggest bottleneck.
    // The loading is done with decreasing granularity for D for better memory bandwidth.
    // __floats2bfloat162_rn(float, float) is CUDA-only; HIP only has the
    // float2-argument __float22bfloat162_rn, which both targets provide.
    const nv_bfloat162 scale_bf16 = __float22bfloat162_rn(make_float2(scale, scale));
#pragma unroll
    for (int stride_k : {warp_size, warp_size/2, warp_size/4, warp_size/8}) {
        const int k0_start  = stride_k == warp_size ? 0 : DKQ/2 - (DKQ/2) % (2*stride_k);
        const int k0_stop   =                             DKQ/2 - (DKQ/2) % (1*stride_k);
        const int stride_jc = warp_size / stride_k;

        if (k0_start == k0_stop) {
            continue;
        }

#pragma unroll
        for (int jc0 = 0; jc0 < ncols; jc0 += nwarps*stride_jc) {
            const int jc = jc0 + threadIdx.y*stride_jc + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

            if (jc0 + nwarps*stride_jc > ncols && jc >= ncols) {
                break;
            }

            const int j = jc / ncols2;
            const int c = jc % ncols2;

            if ((ncols1 == 1 || jt*ncols1 + j < int(ne01.z)) && (ncols2 == 1 || zt_gqa*ncols2 + c < gqa_ratio)) {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    const float2 tmp = Q_f2[(jt*ncols1 + j)*stride_Q1 + c*stride_Q2 + k];
                    tile_Q[jc*stride_tile_Q + k] = scale_bf16 * __float22bfloat162_rn(tmp);
                }
            } else {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    tile_Q[jc*stride_tile_Q + k] = __float22bfloat162_rn(make_float2(0.0f, 0.0f));
                }
            }
        }
    }

    __syncthreads();

    if (Q_in_reg) {
        const int j0 = (threadIdx.y / np) * cols_per_warp;

#pragma unroll
        for (int k0 = 0; k0 < DKQ/2; k0 += T_B_KQ::J) {
            load_ldmatrix(Q_B[k0/T_B_KQ::J], tile_Q + j0*stride_tile_Q + k0, stride_tile_Q);
        }
    }

    __syncthreads();

    int kb0 = kb0_start;

    // kb0_start is always < kb0_stop so the last iter can be executed unconditionally.
    if constexpr (ncols2 == 1 || use_sparse) {
        constexpr bool oob_check = true;
        for (; kb0 < kb0_stop-1; ++kb0) {
            constexpr bool last_iter = false;
            constexpr int  k_VKQ_sup = nbatch_fa;
            flash_attn_ext_bf16_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
                (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
                 KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup);
        }
        constexpr bool last_iter = true;
        const     int  k_VKQ_sup = ne11 - kb0*nbatch_fa;
        flash_attn_ext_bf16_iter
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check,
              T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
            (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
             KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup);
    } else {
        constexpr bool oob_check = false;
        for (; kb0 < kb0_stop-1; ++kb0) {
            constexpr bool last_iter = false;
            constexpr int  k_VKQ_sup = nbatch_fa;
            flash_attn_ext_bf16_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
                (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
                 KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup);
        }
        constexpr bool last_iter = true;
        constexpr int  k_VKQ_sup = nbatch_fa;
        flash_attn_ext_bf16_iter
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check,
             T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
            (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
             KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup);
    }

    // Finally, sum up partial KQ rowsums.
    {
#if defined(AMD_MFMA_AVAILABLE)
        // The partial sums are spread across 4 threads (wavefront64, 16 cols).
        constexpr int offset_first = 32;
        constexpr int offset_last  = 16;
#elif defined(AMD_WMMA_AVAILABLE)
        // The partial sums are spread across 2 threads.
        constexpr int offset_first = 16;
        constexpr int offset_last  = 16;
#else // Ampere
        constexpr int offset_first = 2;
        constexpr int offset_last  = 1;
#endif // defined(AMD_MFMA_AVAILABLE)
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], offset, warp_size);
            }
        }
    }

    // If attention sinks are used, potentially re-scale if KQ_max is small.
    // Also add the sink as a value to KQ_rowsum, this is done after synchronization of KQ_rowsum
    //     so it's being done unconditionally for every thread.
    if (!is_fixup && (np == 1 || threadIdx.y % np == 0) && sinks_f) {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const int jc = (threadIdx.y/np)*cols_per_warp + T_C_KQ::get_i(2*col);
            const float sink = sinks_f[jc % ncols2];

            const float KQ_max_new = fmaxf(KQ_max[col], sink);
            const float KQ_max_diff = KQ_max[col] - KQ_max_new;
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new;

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            const float KQ_max_add = expf(sink - KQ_max_new);
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_max_add;
        }

        // T_C_VKQ is always float here: same KQ_idx derivation as
        // flash_attn_ext_bf16_iter's own VKQ rescale.
#pragma unroll
        for (int i = 0; i < DV/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                constexpr int KQ_idx = 0;
#else
                const int KQ_idx = (l/2) % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                VKQ_C[i].x[l] *= KQ_max_scale[KQ_idx];
            }
        }
    }

    // Combine VKQ accumulator values if np > 1.
    // It's also faster to do small writes to shared memory, then large write to VRAM than to do small writes to VRAM.
    // So also write VKQ accumulators to shared memory in column-major format if np == 1.

    constexpr int tile_stride = nbatch_combine + 4;
    static_assert((DV/2) % nbatch_combine == 0, "bad nbatch_combine");
    constexpr int nbatch_combine_f32 = nbatch_combine/2;
    static_assert((DV/2) % nbatch_combine_f32 == 0, "bad nbatch_combine");

    // jc_cwm = jc combine write meta
    // KQ_cmr = KQ combine max rowsum
    // Use the 16 bytes of padding in each Q column to store the meta data: KQ max, KQ rowsum, KQ max scale.
    // This section only touches KQ_max/KQ_rowsum metadata (not VKQ_C's
    // values), so it is unaffected by T_C_VKQ's element type.
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(0);
        const float2 KQ_cmr = make_float2(KQ_max[0], KQ_rowsum[0]);
        const bool thread_should_write = threadIdx.x / 16 < cols_per_thread;
#else // Ampere
        // On Ampere, T_C_VKQ is the wide tile<16,16,float> shape, whose
        // get_i(l) (`((l/2)%2)*8 + threadIdx.x/4`) only covers 8 of its 16
        // rows per `l` value: `l` in {0,1} addresses rows 0-7, `l` in {2,3}
        // rows 8-15. A single write using `l = threadIdx.x % 4` gated by
        // `threadIdx.x % 4 < cols_per_thread` (2) only ever selects `l` in
        // {0,1} - rows 8-15 of every warp's 16-column range would never get
        // their KQ_rowsum/KQ_max written, later read back as a leftover 0
        // and divided into inf/-inf. Loop over `col` with `l = 2*col`
        // (matching this file's own `T_C_KQ::get_i(2*col)` column
        // derivation, e.g. the sinks rescale above) so both halves get
        // written, from one representative thread per `threadIdx.x/4` group.
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(2*col);
        const float2 KQ_cmr = make_float2(KQ_max[col], KQ_rowsum[col]);
        const bool thread_should_write = threadIdx.x % 4 == 0;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)

        if (((!needs_fixup && !is_fixup) || np > 1) && thread_should_write) {
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && thread_should_write) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && thread_should_write) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    }

    if (np > 1) {
        constexpr int nmeta = np*cols_per_warp >= warp_size ? np*cols_per_warp/warp_size : 1;

        float KQ_cmn;
        float KQ_cms[nmeta];
        float KQ_crs;

        const int jc_meta = threadIdx.y*cols_per_warp + (np*cols_per_warp < warp_size ? threadIdx.x % (np*cols_per_warp) : threadIdx.x);
        float2 * const meta_ptr = ((float2 *) tile_Q) + jc_meta*(tile_stride/2) + nbatch_combine/2;

        if (threadIdx.y % np == 0) {
            // Combine the meta data for parallel warps via shared memory.
            float2 meta[nmeta];
#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                meta[imeta] = meta_ptr[imeta * warp_size * tile_stride/2];
            }

            KQ_cmn = meta[0].x; // KQ combine max new, max between all parallel warps.
#pragma unroll
            for (int imeta = 1; imeta < nmeta; ++imeta) {
                KQ_cmn = fmaxf(KQ_cmn, meta[imeta].x);
            }
#pragma unroll
            for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
                if (offset < warp_size) {
                    KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFF, KQ_cmn, offset, warp_size));
                }
            }

#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                KQ_cms[imeta] = expf(meta[imeta].x - KQ_cmn);
            }

            KQ_crs = KQ_cms[0]*meta[0].y; // KQ combine rowsum, scaled sum of all parallel warps.
#pragma unroll
            for (int imeta = 1; imeta < nmeta; ++imeta) {
                KQ_crs += KQ_cms[imeta]*meta[imeta].y;
            }
#pragma unroll
            for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
                if (offset < warp_size) {
                    KQ_crs += __shfl_xor_sync(0xFFFFFFFF, KQ_crs, offset, warp_size);
                }
            }
        }

        __syncthreads();

        if (threadIdx.y % np == 0) {
            // Write back combined meta data:
#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                if (np*cols_per_warp >= warp_size || threadIdx.x < np*cols_per_warp) {
                    // Combined KQ max scale + rowsum.
                    meta_ptr[imeta * warp_size * tile_stride/2] = make_float2(KQ_cms[imeta], KQ_crs);
                }
            }

            // Combined KQ max + rowsum.
            static_assert(cols_per_warp <= warp_size);
            if (needs_fixup && (cols_per_warp == warp_size || threadIdx.x < cols_per_warp)) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
            }
            if (is_fixup && (cols_per_warp == warp_size || threadIdx.x < cols_per_warp)) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
            }
        }
    }

#pragma unroll
    for (int k00 = 0; k00 < DV/2; k00 += nbatch_combine_f32) {
        // T_C_VKQ is always float (see this file's module doc): stage it
        // in shared memory as float too. A bf16 stage would round the
        // not-yet-normalized accumulator once more before the final
        // rounding of the output. Each row holds nbatch_combine words, so
        // a float stage covers half as many columns per pass.
        const int j0 = threadIdx.y*cols_per_warp;
        float * tile_Q_f = (float *) tile_Q;
#pragma unroll
        for (int k1 = 0; k1 < nbatch_combine_f32; k1 += T_C_VKQ::J/2) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                const int j = j0 + T_C_VKQ::get_i(l);
                const int k = 2*k1 + T_C_VKQ::get_j(l);

                tile_Q_f[j*tile_stride + k] = VKQ_C[(k00 + k1)/(T_C_VKQ::J/2)].x[l];
            }
        }

        __syncthreads();

        if (np == 1 || threadIdx.y % np == 0) {
            // The first 2*2*gridDim.x*ncols floats in dstk_fixup are for storing max. values and row sums.
            // The values after that are for the partial results of the individual blocks.
            float2 * dstk_fixup_data = dstk_fixup + gridDim.x*(2*ncols) + blockIdx.x*(ncols*(DV/2));

#pragma unroll
            for (int stride_k : {warp_size, warp_size/2, warp_size/4, warp_size/8}) {
                const int k0_start  = stride_k == warp_size ? 0 : nbatch_combine_f32 - nbatch_combine_f32 % (2*stride_k);
                const int k0_stop   =                             nbatch_combine_f32 - nbatch_combine_f32 % (1*stride_k);
                const int stride_jc = warp_size / stride_k;

                if (k0_start == k0_stop) {
                    continue;
                }

#pragma unroll
                for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += (nwarps/np)*stride_jc) {
                    const int jc_dst = jc0_dst + (threadIdx.y/np)*stride_jc + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                    if (jc0_dst + (nwarps/np)*stride_jc > ncols && jc_dst >= ncols) {
                        break;
                    }

                    const int jc_tile_K = (jc_dst/cols_per_warp)*(np*cols_per_warp) + jc_dst % cols_per_warp;

                    const int j_dst = jc_dst / ncols2;
                    const int c_dst = jc_dst % ncols2;

                    if (!is_fixup && ((ncols1 > 1 && jt*ncols1 + j_dst >= int(ne01.z)) || (ncols2 > 1 && zt_gqa*ncols2 + c_dst >= gqa_ratio))) {
                        continue;
                    }

                    const float * meta_j = (const float *) tile_Q + jc_tile_K*tile_stride + nbatch_combine;
#pragma unroll
                    for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                        const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                        float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                        for (int ip = 0; ip < np; ++ip) {
                            const float KQ_crs = np == 1 ? 1.0f : meta_j[ip*cols_per_warp * tile_stride + 0];
                            const float2 dstk_val_add = ((const float2 *) tile_Q_f)[((jc_tile_K + ip*cols_per_warp) * tile_stride)/2 + k];
                            dstk_val.x += dstk_val_add.x*KQ_crs;
                            dstk_val.y += dstk_val_add.y*KQ_crs;
                        }

                        if (!needs_fixup && !is_fixup) {
                            const float KQ_rowsum_j = meta_j[1];
                            dstk_val.x /= KQ_rowsum_j;
                            dstk_val.y /= KQ_rowsum_j;
                        }

                        if (is_fixup) {
                            dstk_fixup_data[jc_dst*(DV/2) + k00 + k] = dstk_val;
                        } else {
                            dstk[((jt*ncols1 + j_dst)*ne02 + c_dst)*(DV/2) + k00 + k] = dstk_val;
                        }
                    }
                }
            }
        }
        if (np > 1) {
            __syncthreads();
        }
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_h2, V_h2, mask_h, indices, sinks_f, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02, gqa_ratio,
        stride_Q1, stride_Q2, stride_K, stride_V, stride_mask,
        jt, kb0_start, kb0_stop);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

// __device__, not __global__: CUDA/HIP forbid calling a __global__ kernel
// directly from another __global__ function, and DECL_FATTN_MMA_BF16_CASE's
// extern "C" wrapper calls this directly rather than through a <<<>>>
// launch. __launch_bounds__ moves to that wrapper instead, since it is only
// meaningful on an actual kernel entry point. Unlike fattn_mma_f16.cuh's
// flash_attn_ext_f16, the `use_sparse`/DKQ==192/Turing/Volta early-exit
// checks are dropped: `use_sparse` is always `false` per
// `DECL_FATTN_MMA_BF16_CASE` (making those checks unconditionally
// unreachable already in the f16 kernel too), DKQ==192 is not one of
// Crane's instantiated head dims, and bf16 MMA never runs on Turing/Volta
// in the first place (see this file's module doc). The AMD tile-size
// guards are kept: they are real constraints on the bf16 MMA/WMMA
// overloads this kernel's mma_tile_sizes resolves to.
template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, bool V_is_K_view, bool use_sparse>
static __device__ __forceinline__ void flash_attn_ext_bf16(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_sync(); // TODO optimize placement
#if defined(FLASH_ATTN_AVAILABLE) && (defined(AMPERE_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE))
    const char * GGML_CUDA_RESTRICT Q              = Q_ptr;
    const char * GGML_CUDA_RESTRICT K              = K_ptr;
    const char * GGML_CUDA_RESTRICT V              = V_ptr;
    const char * GGML_CUDA_RESTRICT mask           = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks          = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max         = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst            = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta       = dst_meta_ptr;

#if defined(AMD_WMMA_AVAILABLE)
    if (ncols1*ncols2 < 16 || ncols2 == 1 || DKQ > 256) {
        NO_DEVICE_CODE;
        return;
    }
#endif // defined(AMD_WMMA_AVAILABLE)

#if defined(AMD_MFMA_AVAILABLE)
    if (ncols1*ncols2 < 16 || DKQ > 256) {
        NO_DEVICE_CODE;
        return;
    }
#endif // defined(AMD_MFMA_AVAILABLE)

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int ncols     = ncols1 * ncols2;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int nthreads  = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols);
    constexpr int nwarps    = nthreads / warp_size;

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_K    = nb11 / sizeof(nv_bfloat162);
    const int stride_mask = nb31 / sizeof(half);

    const int stride_V = V_is_K_view ? stride_K : nb21 / sizeof(nv_bfloat162);

    const int iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    const int iter_j     = (ne01.z    + (ncols1    - 1)) / ncols1;
    const int iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    // kbc == k block continuous, current index in continuous ijk space.
    int       kbc      = int64_t(blockIdx.x + 0)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;
    const int kbc_stop = int64_t(blockIdx.x + 1)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;

    // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
    // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).
    // In the most general case >2 seams can fall into the same tile.

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
        const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
        const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
        const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
        const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

        const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

        const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
        const nv_bfloat162 * K_h2 = (const nv_bfloat162 *) (K + nb13*sequence + nb12*z_KV);
        const half   * mask_h = ncols2 == 1 && !mask ? nullptr :
            (const half *) (mask + nb33*(sequence % ne33));
        float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);

        const nv_bfloat162 * V_h2 = V_is_K_view ? K_h2 : (const nv_bfloat162 *) (V + nb23*sequence + nb22*z_KV);
        const float * sinks_f = sinks ? (const float *) sinks + zt_Q : nullptr;

        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

        if (KV_max) {
            kb0_stop = min(kb0_stop, KV_max[sequence*iter_j + jt] / nbatch_fa);
        }
        constexpr bool is_fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false; // CUDA block is working on an entire tile.
            flash_attn_ext_bf16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, mask_h, nullptr, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop);
        } else {
            constexpr bool needs_fixup = true; // CUDA block is missing the beginning of a tile.
            flash_attn_ext_bf16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, mask_h, nullptr, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index.
    const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
    const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
    const nv_bfloat162 * K_h2 = (const nv_bfloat162 *) (K + nb13*sequence + nb12*z_KV);
    const half   * mask_h = ncols2 == 1 && !mask ? nullptr :
        (const half *) (mask + nb33*(sequence % ne33));
    float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);

    const nv_bfloat162 * V_h2 = V_is_K_view ? K_h2 : (const nv_bfloat162 *) (V + nb23*sequence + nb22*z_KV);
    const float * sinks_f = sinks ? (const float *) sinks + zt_Q : nullptr;

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

    if (KV_max) {
        kb0_stop = min(kb0_stop, KV_max[sequence*iter_j + jt] / nbatch_fa);
    }

    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    constexpr bool needs_fixup = false;
    flash_attn_ext_bf16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup>
        (Q_f2, K_h2, V_h2, mask_h, nullptr, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
         ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop);
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && (defined(AMPERE_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE))
}

#define DECL_FATTN_MMA_BF16_CASE(DKQ_, DV_, ncols1_, ncols2_)                                          \
    extern "C" __global__                                                                              \
    __launch_bounds__(                                                                                 \
        ggml_cuda_fattn_mma_get_nthreads(DKQ_, DV_, (ncols1_) * (ncols2_)),                             \
        ggml_cuda_fattn_mma_get_occupancy(DKQ_, DV_, (ncols1_) * (ncols2_)))                            \
    void crane_fattn_mma_bf16_d##DKQ_##_v##DV_##_c##ncols1_##_s##ncols2_(                              \
        const char * Q, const char * K, const char * V,                                                \
        const char * mask, const char * sinks, const int * KV_max,                                     \
        float * dst, float2 * dst_meta,                                                                \
        float scale, float max_bias, float m0, float m1,                                               \
        uint32_t n_head_log2, float logit_softcap,                                                     \
        int32_t ne00, uint3 ne01, int32_t ne02, int32_t ne03,                                          \
        int32_t nb01, int32_t nb02, int32_t nb03,                                                       \
        int32_t ne10, int32_t ne11, int32_t ne12, int32_t ne13,                                         \
        int32_t nb11, int32_t nb12, int64_t nb13,                                                       \
        int32_t nb21, int32_t nb22, int64_t nb23,                                                       \
        int32_t ne31, int32_t ne32, int32_t ne33,                                                       \
        int32_t nb31, int32_t nb32, int64_t nb33)                                                       \
    {                                                                                                    \
        flash_attn_ext_bf16<DKQ_, DV_, ncols1_, ncols2_, false, false, false>(                          \
            Q, K, V, mask, sinks, KV_max, dst, dst_meta,                                                \
            scale, max_bias, m0, m1, n_head_log2, logit_softcap,                                        \
            ne00, ne01, ne02, ne03, nb01, nb02, nb03,                                                    \
            ne10, ne11, ne12, ne13, nb11, nb12, nb13,                                                    \
            nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);                                       \
    }
