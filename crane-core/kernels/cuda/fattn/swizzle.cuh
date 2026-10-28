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
// Vendored from llama.cpp's ggml/src/ggml-cuda/fattn-swizzle.cuh
// SPDX-License-Identifier: MIT
/**
 * Bank-conflict-avoiding XOR swizzle for the MMA kernel's K/V shared-memory
 * tiles (Turing+ NVIDIA only, see `TURING_MMA_AVAILABLE` in
 * `crane_fattn_shim.cuh`; always disabled on AMD). Upstream's source file
 * is named `fattn-swizzle.cuh`; renamed here to `swizzle.cuh` to match
 * this directory's naming convention. Used by `fattn_mma_f16.cuh`.
 *
 * Drops the host-callable `enabled(nbatch_2, cc)`/`tile_stride(nbatch_2,
 * cc)` overloads: both were only reachable from llama.cpp's host dispatch
 * (`ggml_cuda_flash_attn_ext_mma_f16_case`, not vendored), never from the
 * kept kernel code, which always goes through the `constexpr __device__`
 * overloads below instead.
 */
#pragma once

#include "crane_fattn_shim.cuh"
#include "mma.cuh"


namespace ggml_cuda_fattn_smem_swizzle {

static __host__ __device__ constexpr bool bank_aligned(const int nbatch_2) {
    return nbatch_2 >= 32 && nbatch_2 % 32 == 0;
}

static __device__ constexpr bool enabled(const int nbatch_2) {
#if defined(TURING_MMA_AVAILABLE)
    return bank_aligned(nbatch_2);
#else
    GGML_UNUSED(nbatch_2);
    return false;
#endif // defined(TURING_MMA_AVAILABLE)
}

static __device__ constexpr int tile_stride(const int nbatch_2) {
    return enabled(nbatch_2) ? nbatch_2 : nbatch_2 + 4;
}

// Swizzled byte offset for tile element (row, col_h2), same map used for writes and reads.
template<int stride_h2>
static __device__ __forceinline__ int bytes_rc(const int row, const int col_h2) {
    static_assert(bank_aligned(stride_h2), "swizzled tile needs a stride that is a multiple of 32");
    return ((row * stride_h2 + col_h2) * (int) sizeof(half2)) ^ ((row & 7) << 4);
}

// ldmatrix.x4 via 64-bit generic pointer.
static __device__ __forceinline__ void ldmatrix_x4(int * xi, const half2 * addr) {
#if defined(TURING_MMA_AVAILABLE)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
        : "l"(addr));
#else
    GGML_UNUSED_VARS(xi, addr);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}

static __device__ __forceinline__ void ldmatrix_x4_trans(int * xi, const half2 * addr) {
#if defined(TURING_MMA_AVAILABLE)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
        : "l"(addr));
#else
    GGML_UNUSED_VARS(xi, addr);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}

// Per-lane swizzled address for one tile<16, 8, half2> ldmatrix: 16 rows, 4 half2 columns per lane.
template<int stride_h2>
static __device__ __forceinline__ const half2 * lane_addr(
        const half2 * tile_base, const int base_row, const int base_col_h2, const int I, const int J) {
    static_assert(bank_aligned(stride_h2), "swizzled tile needs a stride that is a multiple of 32");
    const int lane_row = threadIdx.x % I;
    const int lane_col = (threadIdx.x / I) * (J / 2);
    uint32_t byte_off = (uint32_t) ((base_row + lane_row)*stride_h2 + base_col_h2 + lane_col) * (uint32_t) sizeof(half2);
    byte_off ^= (uint32_t) (((base_row + lane_row) & 7) << 4);
    return (const half2 *) ((const char *) tile_base + byte_off);
}

template<int stride_h2, bool swz, typename TileT>
static __device__ __forceinline__ void load_ldmatrix(
        TileT & t, const half2 * tile_base, const int base_row, const int base_col_h2) {
    if constexpr (swz) {
        static_assert(std::is_same_v<TileT, ggml_cuda_mma::tile<16, 8, half2>>,
            "the swizzled layout is only supported for tile<16, 8, half2>");
        ldmatrix_x4((int *) t.x, lane_addr<stride_h2>(tile_base, base_row, base_col_h2, TileT::I, TileT::J));
    } else {
        ggml_cuda_mma::load_ldmatrix(t, tile_base + base_row*stride_h2 + base_col_h2, stride_h2);
    }
}

template<int stride_h2, bool swz, typename TileT>
static __device__ __forceinline__ void load_ldmatrix(TileT & t, const half2 * tile_base, const int off_h2) {
    if constexpr (swz) {
        load_ldmatrix<stride_h2, swz>(t, tile_base, off_h2 / stride_h2, off_h2 % stride_h2);
    } else {
        ggml_cuda_mma::load_ldmatrix(t, tile_base + off_h2, stride_h2);
    }
}

template<int stride_h2, bool swz, typename TileT>
static __device__ __forceinline__ void load_ldmatrix_trans(
        TileT & t, const half2 * tile_base, const int base_row, const int base_col_h2) {
    if constexpr (swz) {
        static_assert(std::is_same_v<TileT, ggml_cuda_mma::tile<16, 8, half2>>,
            "the swizzled layout is only supported for tile<16, 8, half2>");
        ldmatrix_x4_trans((int *) t.x, lane_addr<stride_h2>(tile_base, base_row, base_col_h2, TileT::I, TileT::J));
    } else {
        ggml_cuda_mma::load_ldmatrix_trans(t, tile_base + base_row*stride_h2 + base_col_h2, stride_h2);
    }
}

template<int stride_h2, bool swz, typename TileT>
static __device__ __forceinline__ void load_ldmatrix_trans(TileT & t, const half2 * tile_base, const int off_h2) {
    if constexpr (swz) {
        load_ldmatrix_trans<stride_h2, swz>(t, tile_base, off_h2 / stride_h2, off_h2 % stride_h2);
    } else {
        ggml_cuda_mma::load_ldmatrix_trans(t, tile_base + off_h2, stride_h2);
    }
}

} // namespace ggml_cuda_fattn_smem_swizzle
