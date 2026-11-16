// SPDX-License-Identifier: MIT
//! Template instantiations of the tensor-core flash-attention kernel at
//! head_dim=512 (Gemma4): one `extern "C"` entry point per (ncols1, ncols2)
//! tile-size variant (`crane_fattn_mma_f16_d512_v512_c<ncols1>_s<ncols2>`).
//! Unlike `fattn_mma_hd{64,128,256}.cu`'s full ncols-sweep, this mirrors
//! upstream llama.cpp's own narrower instance list for DKQ=DV=512:
//! `select_mma_ncols` (in `ops::fused_ops::fattn`) rejects `ncols2==1` for
//! head_dim=512, so only `ncols2 in {2, 4, 8}` is instantiated here.
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

// Bounds-checked-tail counterparts (see DECL_FATTN_MMA_F16_TAIL_CASE's doc
// comment) of every case above: head_dim=512 never instantiates ncols2 == 1
// (see this file's module doc), so every case here needs one.
DECL_FATTN_MMA_F16_TAIL_CASE(512, 512,  4,  2);
DECL_FATTN_MMA_F16_TAIL_CASE(512, 512,  8,  2);
DECL_FATTN_MMA_F16_TAIL_CASE(512, 512, 16,  2);
DECL_FATTN_MMA_F16_TAIL_CASE(512, 512, 32,  2);
DECL_FATTN_MMA_F16_TAIL_CASE(512, 512,  2,  4);
DECL_FATTN_MMA_F16_TAIL_CASE(512, 512,  4,  4);
DECL_FATTN_MMA_F16_TAIL_CASE(512, 512,  8,  4);
DECL_FATTN_MMA_F16_TAIL_CASE(512, 512, 16,  4);
DECL_FATTN_MMA_F16_TAIL_CASE(512, 512,  1,  8);
DECL_FATTN_MMA_F16_TAIL_CASE(512, 512,  2,  8);
DECL_FATTN_MMA_F16_TAIL_CASE(512, 512,  4,  8);
DECL_FATTN_MMA_F16_TAIL_CASE(512, 512,  8,  8);
