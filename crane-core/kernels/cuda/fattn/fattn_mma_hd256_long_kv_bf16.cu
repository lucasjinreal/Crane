// SPDX-License-Identifier: MIT
//! The 32- and 64-column head_dim=256 tile sizes of `fattn_mma_hd256_bf16.cu`,
//! built with the long-KV tile config (see `CRANE_FATTN_LONG_KV` in
//! `fattn_mma_bf16.cuh`). Same symbol names as that file: the host loads
//! these from a separate module.

#define CRANE_FATTN_LONG_KV

#include "fattn_mma_bf16.cuh"

DECL_FATTN_MMA_BF16_CASE(256, 256, 32,  1);
DECL_FATTN_MMA_BF16_CASE(256, 256, 16,  2);
DECL_FATTN_MMA_BF16_CASE(256, 256,  8,  4);
DECL_FATTN_MMA_BF16_CASE(256, 256,  4,  8);
DECL_FATTN_MMA_BF16_CASE(256, 256,  2, 16);
DECL_FATTN_MMA_BF16_CASE(256, 256, 64,  1);
DECL_FATTN_MMA_BF16_CASE(256, 256, 32,  2);
DECL_FATTN_MMA_BF16_CASE(256, 256, 16,  4);
DECL_FATTN_MMA_BF16_CASE(256, 256,  8,  8);
DECL_FATTN_MMA_BF16_CASE(256, 256,  4, 16);

// Bounds-checked-tail counterparts (see DECL_FATTN_MMA_BF16_TAIL_CASE's doc
// comment) of every ncols2 > 1 case above, for unaligned seq_kv.
DECL_FATTN_MMA_BF16_TAIL_CASE(256, 256, 16,  2);
DECL_FATTN_MMA_BF16_TAIL_CASE(256, 256,  8,  4);
DECL_FATTN_MMA_BF16_TAIL_CASE(256, 256,  4,  8);
DECL_FATTN_MMA_BF16_TAIL_CASE(256, 256,  2, 16);
DECL_FATTN_MMA_BF16_TAIL_CASE(256, 256, 32,  2);
DECL_FATTN_MMA_BF16_TAIL_CASE(256, 256, 16,  4);
DECL_FATTN_MMA_BF16_TAIL_CASE(256, 256,  8,  8);
DECL_FATTN_MMA_BF16_TAIL_CASE(256, 256,  4, 16);
