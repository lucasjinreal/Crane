// SPDX-License-Identifier: MIT
//! Template instantiations of the tile flash-attention kernel (the MMA
//! kernel's fallback) at head_dim 64/128/256/512: one `extern "C"` entry
//! point per head_dim (`crane_fattn_tile_f16_d<hd>_v<hd>_c16_s1`). Phase 1
//! instantiates exactly one (ncols1, ncols2) = (16, 1) batch-size tier per
//! head_dim rather than porting upstream's full ncols sweep; see
//! `fattn_tile.cuh`'s module doc for why that's enough for this fallback
//! path.

#include "fattn_tile.cuh"

DECL_FATTN_TILE_CASE(64, 64, 16, 1);
DECL_FATTN_TILE_CASE(128, 128, 16, 1);
DECL_FATTN_TILE_CASE(256, 256, 16, 1);
DECL_FATTN_TILE_CASE(512, 512, 16, 1);
