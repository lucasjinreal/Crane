// Puts the active candle backend into the extern prelude as `candle_core` /
// `candle_nn` / `candle_transformers`, so existing call sites don't change
// per backend. Must be `include!`d at the crate root (not a `mod`) for the
// prelude effect to apply. `sycl` wins if both `sycl` and `rocm` are set.
//
// This is `pub`, so downstream crates (crane-serve, crane) reach the active
// backend as `crane_core::candle_core` / `::candle_nn` / `::candle_transformers`
// instead of depending on a candle-core package themselves — that's what
// keeps the backend choice a single knob in crane-core's own Cargo features.
// The extern-prelude effect above is local to this crate, so a downstream
// module still needs its own `use crane_core::candle_core;` (etc.) to use
// the unqualified `candle_core::` path.

// Intel oneAPI/SYCL fork. Delete this block (and the `candle-*-sycl`
// workspace deps) once SYCL support merges into candle main — at that
// point `sycl` just becomes another feature on the plain `candle-core` dep
// below, same as `cuda`/`metal`.
#[cfg(feature = "sycl")]
pub extern crate candle_core_sycl as candle_core;
#[cfg(feature = "sycl")]
pub extern crate candle_nn_sycl as candle_nn;
#[cfg(feature = "sycl")]
pub extern crate candle_transformers_sycl as candle_transformers;

// AMD ROCm/HIP fork. Delete this block (and the `candle-*-rocm` workspace
// deps) once ROCm support merges into candle main, same as the SYCL note
// above.
#[cfg(all(feature = "rocm", not(feature = "sycl")))]
pub extern crate candle_core_rocm as candle_core;
#[cfg(all(feature = "rocm", not(feature = "sycl")))]
pub extern crate candle_nn_rocm as candle_nn;
#[cfg(all(feature = "rocm", not(feature = "sycl")))]
pub extern crate candle_transformers_rocm as candle_transformers;

#[cfg(not(any(feature = "sycl", feature = "rocm")))]
pub extern crate candle_core;
#[cfg(not(any(feature = "sycl", feature = "rocm")))]
pub extern crate candle_nn;
#[cfg(not(any(feature = "sycl", feature = "rocm")))]
pub extern crate candle_transformers;

// Flat re-exports of candle's foundational, backend-agnostic types, so
// outside crates (crane-serve, crane, tests) can write `crane_core::Tensor`
// instead of reaching into `crane_core::candle_core::Tensor`, and ideally
// never need a bare `use crane_core::candle_core;` at all. `candle_core`
// etc. above still cover anything not listed here (ops modules, GGUF
// internals, backend-specific items) — when outside code needs one of
// those regularly, add it here rather than importing the module wholesale.
// No name below collides with an existing crane-core root item.
pub use candle_core::quantized::{GgmlDType, gguf_file};
pub use candle_core::utils::{cuda_is_available, metal_is_available};
pub use candle_core::{D, DType, Device, Error, Result, Tensor, bail};
pub use candle_nn::ops::softmax_last_dim;
pub use candle_transformers::generation::LogitsProcessor;
