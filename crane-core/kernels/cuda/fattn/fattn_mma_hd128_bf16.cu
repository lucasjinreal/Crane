// SPDX-License-Identifier: MIT
//! Template instantiations of the native-bf16 tensor-core flash-attention
//! kernel at head_dim=128: one `extern "C"` entry point per (ncols1, ncols2)
//! tile-size variant (`crane_fattn_mma_bf16_d128_v128_c<ncols1>_s<ncols2>`).
//! See `fattn_mma_hd64_bf16.cu`'s module doc for why `ncols1*ncols2 == 8` is
//! omitted here (unlike the f16 sweep in `fattn_mma_hd128.cu`).

#include "fattn_mma_bf16.cuh"

DECL_FATTN_MMA_BF16_CASE(128, 128, 16,  1);
DECL_FATTN_MMA_BF16_CASE(128, 128,  8,  2);
DECL_FATTN_MMA_BF16_CASE(128, 128,  4,  4);
DECL_FATTN_MMA_BF16_CASE(128, 128,  2,  8);
DECL_FATTN_MMA_BF16_CASE(128, 128,  1, 16);
DECL_FATTN_MMA_BF16_CASE(128, 128, 32,  1);
DECL_FATTN_MMA_BF16_CASE(128, 128, 16,  2);
DECL_FATTN_MMA_BF16_CASE(128, 128,  8,  4);
DECL_FATTN_MMA_BF16_CASE(128, 128,  4,  8);
DECL_FATTN_MMA_BF16_CASE(128, 128,  2, 16);
DECL_FATTN_MMA_BF16_CASE(128, 128, 64,  1);
DECL_FATTN_MMA_BF16_CASE(128, 128, 32,  2);
DECL_FATTN_MMA_BF16_CASE(128, 128, 16,  4);
DECL_FATTN_MMA_BF16_CASE(128, 128,  8,  8);
DECL_FATTN_MMA_BF16_CASE(128, 128,  4, 16);
