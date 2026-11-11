// SPDX-License-Identifier: MIT
//! The 64-column head_dim=256 tile sizes of `fattn_mma_hd256.cu`, built with
//! the long-KV tile config (see `CRANE_FATTN_LONG_KV` in
//! `fattn_mma_f16.cuh`). Same symbol names as that file: the host loads
//! these from a separate module. ROCm only, so the `.hip` extension keeps
//! the CUDA build from compiling it.

#define CRANE_FATTN_LONG_KV

#include "fattn_mma_f16.cuh"

DECL_FATTN_MMA_F16_CASE(256, 256, 32,  2);
DECL_FATTN_MMA_F16_CASE(256, 256, 16,  4);
DECL_FATTN_MMA_F16_CASE(256, 256,  8,  8);
DECL_FATTN_MMA_F16_CASE(256, 256,  4, 16);

DECL_FATTN_MMA_F16_TAIL_CASE(256, 256, 32,  2);
DECL_FATTN_MMA_F16_TAIL_CASE(256, 256, 16,  4);
DECL_FATTN_MMA_F16_TAIL_CASE(256, 256,  8,  8);
DECL_FATTN_MMA_F16_TAIL_CASE(256, 256,  4, 16);
