// SPDX-License-Identifier: MIT
//! Template instantiations of the native-bf16 tensor-core flash-attention
//! kernel at head_dim=64: one `extern "C"` entry point per (ncols1, ncols2)
//! tile-size variant (`crane_fattn_mma_bf16_d64_v64_c<ncols1>_s<ncols2>`).
//! Unlike `fattn_mma_hd64.cu`'s f16 sweep, `ncols1*ncols2 == 8` is omitted:
//! this kernel never instantiates the narrow (cols_per_warp == 8) tile
//! shape (see `fattn_mma_bf16.cuh`'s module doc), so the smallest `ncols`
//! here is 16. See that file's module doc for what each variant means and
//! `ops::fused_ops::fattn`'s kernel-selection logic for how Rust picks
//! among them at runtime.

#include "fattn_mma_bf16.cuh"

DECL_FATTN_MMA_BF16_CASE(64, 64, 16,  1);
DECL_FATTN_MMA_BF16_CASE(64, 64,  8,  2);
DECL_FATTN_MMA_BF16_CASE(64, 64,  4,  4);
DECL_FATTN_MMA_BF16_CASE(64, 64,  2,  8);
DECL_FATTN_MMA_BF16_CASE(64, 64,  1, 16);
DECL_FATTN_MMA_BF16_CASE(64, 64, 32,  1);
DECL_FATTN_MMA_BF16_CASE(64, 64, 16,  2);
DECL_FATTN_MMA_BF16_CASE(64, 64,  8,  4);
DECL_FATTN_MMA_BF16_CASE(64, 64,  4,  8);
DECL_FATTN_MMA_BF16_CASE(64, 64,  2, 16);
DECL_FATTN_MMA_BF16_CASE(64, 64, 64,  1);
DECL_FATTN_MMA_BF16_CASE(64, 64, 32,  2);
DECL_FATTN_MMA_BF16_CASE(64, 64, 16,  4);
DECL_FATTN_MMA_BF16_CASE(64, 64,  8,  8);
DECL_FATTN_MMA_BF16_CASE(64, 64,  4, 16);
