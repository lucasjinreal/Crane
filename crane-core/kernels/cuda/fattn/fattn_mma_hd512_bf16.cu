// SPDX-License-Identifier: MIT
//! Template instantiations of the native-bf16 tensor-core flash-attention
//! kernel at head_dim=512 (Gemma4): one `extern "C"` entry point per
//! (ncols1, ncols2) tile-size variant
//! (`crane_fattn_mma_bf16_d512_v512_c<ncols1>_s<ncols2>`). Mirrors
//! `fattn_mma_hd512.cu`'s narrower instance list for DKQ=DV=512, minus the
//! `ncols1*ncols2 == 8` combinations (see `fattn_mma_hd64_bf16.cu`'s module
//! doc for why this kernel never instantiates that tile shape).

#include "fattn_mma_bf16.cuh"

DECL_FATTN_MMA_BF16_CASE(512, 512,  8,  2);
DECL_FATTN_MMA_BF16_CASE(512, 512, 16,  2);
DECL_FATTN_MMA_BF16_CASE(512, 512, 32,  2);
DECL_FATTN_MMA_BF16_CASE(512, 512,  4,  4);
DECL_FATTN_MMA_BF16_CASE(512, 512,  8,  4);
DECL_FATTN_MMA_BF16_CASE(512, 512, 16,  4);
DECL_FATTN_MMA_BF16_CASE(512, 512,  2,  8);
DECL_FATTN_MMA_BF16_CASE(512, 512,  4,  8);
DECL_FATTN_MMA_BF16_CASE(512, 512,  8,  8);
