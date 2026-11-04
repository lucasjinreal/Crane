// SPDX-License-Identifier: MIT
//! Template instantiations of the tensor-core flash-attention kernel at
//! head_dim=256: one `extern "C"` entry point per (ncols1, ncols2)
//! tile-size variant (`crane_fattn_mma_f16_d256_v256_c<ncols1>_s<ncols2>`).
//! See `fattn_mma_f16.cuh`'s module doc for what each variant means and
//! `ops::fused_ops::fattn`'s kernel-selection logic for how Rust picks
//! among them at runtime.

#include "fattn_mma_f16.cuh"

DECL_FATTN_MMA_F16_CASE(256, 256,  8,  1);
DECL_FATTN_MMA_F16_CASE(256, 256,  4,  2);
DECL_FATTN_MMA_F16_CASE(256, 256,  2,  4);
DECL_FATTN_MMA_F16_CASE(256, 256,  1,  8);
DECL_FATTN_MMA_F16_CASE(256, 256, 16,  1);
DECL_FATTN_MMA_F16_CASE(256, 256,  8,  2);
DECL_FATTN_MMA_F16_CASE(256, 256,  4,  4);
DECL_FATTN_MMA_F16_CASE(256, 256,  2,  8);
DECL_FATTN_MMA_F16_CASE(256, 256,  1, 16);
DECL_FATTN_MMA_F16_CASE(256, 256, 32,  1);
DECL_FATTN_MMA_F16_CASE(256, 256, 16,  2);
DECL_FATTN_MMA_F16_CASE(256, 256,  8,  4);
DECL_FATTN_MMA_F16_CASE(256, 256,  4,  8);
DECL_FATTN_MMA_F16_CASE(256, 256,  2, 16);
DECL_FATTN_MMA_F16_CASE(256, 256, 64,  1);
DECL_FATTN_MMA_F16_CASE(256, 256, 32,  2);
DECL_FATTN_MMA_F16_CASE(256, 256, 16,  4);
DECL_FATTN_MMA_F16_CASE(256, 256,  8,  8);
DECL_FATTN_MMA_F16_CASE(256, 256,  4, 16);
