// SPDX-License-Identifier: MIT
//! Template instantiations of the tensor-core flash-attention kernel at
//! head_dim=512 (Gemma4): one `extern "C"` entry point per (ncols1, ncols2)
//! tile-size variant (`crane_fattn_mma_f16_d512_v512_c<ncols1>_s<ncols2>`).
//! Unlike `fattn_mma_hd{64,128,256}.cu`'s full ncols-sweep, this mirrors
//! upstream llama.cpp's own narrower instance list for DKQ=DV=512, which
//! has its own GQA constraints (`gqa_opt_applies`, checked in
//! `ops::fused_ops::fattn`'s dispatch) that rule out most ncols2 splits.
//! See `fattn_mma_f16.cuh`'s module doc for what each variant means and
//! `ops::fused_ops::fattn`'s kernel-selection logic for how Rust picks
//! among them at runtime.

#include "fattn_mma_f16.cuh"

DECL_FATTN_MMA_F16_CASE(512, 512,  4,  2);
DECL_FATTN_MMA_F16_CASE(512, 512,  8,  2);
DECL_FATTN_MMA_F16_CASE(512, 512, 16,  2);
DECL_FATTN_MMA_F16_CASE(512, 512, 32,  2);
DECL_FATTN_MMA_F16_CASE(512, 512,  2,  4);
DECL_FATTN_MMA_F16_CASE(512, 512,  4,  4);
DECL_FATTN_MMA_F16_CASE(512, 512,  8,  4);
DECL_FATTN_MMA_F16_CASE(512, 512, 16,  4);
DECL_FATTN_MMA_F16_CASE(512, 512,  1,  8);
DECL_FATTN_MMA_F16_CASE(512, 512,  2,  8);
DECL_FATTN_MMA_F16_CASE(512, 512,  4,  8);
DECL_FATTN_MMA_F16_CASE(512, 512,  8,  8);
