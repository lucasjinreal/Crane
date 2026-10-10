// SPDX-License-Identifier: MIT

//! Shared GGUF quantized-weight loading infrastructure, used by every model
//! that supports loading from a GGUF checkpoint (`hunyuan_dense`, `gemma4`,
//! `qwen3`, `qwen3_5`, `minicpm5`, `minicpmo`).

// GGUF tensor dims/string lengths narrowed in this module are bounds-checked
// against the file's own declared size before the cast, or are u64 values
// that fit losslessly in usize on every supported (64-bit) target.
#![allow(clippy::cast_possible_truncation)]

pub mod extended_gguf;
pub mod gguf_file;
pub mod gguf_metadata;
pub mod iquant;
mod iquant_grids;
pub mod ternary;
#[cfg(test)]
pub(crate) mod test_util;
