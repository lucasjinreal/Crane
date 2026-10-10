// SPDX-License-Identifier: MIT
/**
 * Minimal replacement for llama.cpp's `common.cuh` (1711 lines), providing
 * only what the vendored flash-attention headers under this directory
 * (`mma.cuh`, `cp_async.cuh`, `swizzle.cuh`, `fattn_common.cuh`,
 * `fattn_mma_f16.cuh`, `fattn_tile.cuh`) actually reference: compile-time
 * architecture detection, fast integer division, and a handful of small
 * device-side utility functions. Deliberately excludes `ggml.h`,
 * `ggml_tensor`, `ggml_type`, `ggml_backend_cuda_context`, `convert.cuh`,
 * `vecdotq.cuh`, `launch_fattn`, and ggml's memory-pool allocators. Those
 * belong to the host-side dispatch code llama.cpp's `fattn-*.cuh` files use
 * to build a `ggml_tensor` graph node, which Crane's Rust dispatch layer
 * (`ops/fused_ops/fattn.rs`, `fattn_mma.rs`, `fattn_tile.rs`) replaces
 * entirely rather than porting.
 *
 * `GGML_USE_HIP` is defined here so the vendored files `#if
 * defined(GGML_USE_HIP)` conditionals keep working unmodified.
 *
 * Also intentionally omits `init_fastdiv_values` (the host-side `uint3`
 * packer): nothing in the vendored kernel source calls it, since Crane
 * packs each launch's `ne01` on the Rust side
 * (`ops::fused_ops::fattn::init_fastdiv_values`) before passing it as a
 * kernel argument. Only the device-side consumers (`fastdiv`/`fastmodulo`/
 * `fast_div_modulo`) are needed here.
 */
#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Real CUDA's <cuda_bf16.h> provides both the double-underscore-prefixed
// `__nv_bfloat162` and the unprefixed `nv_bfloat162` alias `mma.cuh` uses;
// HIP's CUDA-compatibility shim (`hip_shim/cuda_bf16.h`, used by hipcc to
// parse CUDA-flavored source) only typedefs the prefixed name, confirmed
// empirically (gfx1201 build: "unknown type name 'nv_bfloat162'"). Safe to
// declare unconditionally: redeclaring an identical type alias is valid
// C++, so this is a no-op under real CUDA where the alias already exists.
using nv_bfloat162 = __nv_bfloat162;

#if defined(__HIPCC__)
#define GGML_USE_HIP
#endif // defined(__HIPCC__)

// candle's rocm backend unconditionally passes `-DRDNA3` to every custom
// kernel it compiles (confirmed empirically: a real gfx1201/RDNA4 build
// still had `RDNA3` defined on the command line), presumably because its
// own kernels never distinguish RDNA3 from RDNA4. This vendored kernel
// family does need that distinction (RDNA4's WMMA intrinsics differ from
// RDNA3's. Confirmed by a real "Cannot select: intrinsic
// %llvm.amdgcn.wmma.f16.16x16x16.f16" backend failure on gfx1201 before
// this fix, from mma.cuh picking the RDNA3 intrinsic branch instead of the
// RDNA4 one. So `__GFX11__`/`__GFX12__`, real compiler-predefined
// architecture macros, not build-system flags, must be the sole source of
// truth here. Undefine first so the externally-passed flag can't win.
#undef RDNA3
#undef RDNA4

#if defined(__HIPCC__) && defined(__GFX12__)
#define RDNA4
#endif // defined(__HIPCC__) && defined(__GFX12__)

#if defined(__HIPCC__) && defined(__GFX11__)
#define RDNA3
#endif // defined(__HIPCC__) && defined(__GFX11__)

#if defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#define RDNA3_5
#endif // defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)

#if defined(RDNA3) && !defined(RDNA3_5)
#define RDNA3_0
#endif // defined(RDNA3) && !defined(RDNA3_5)

#define WARP_SIZE 32

// RDNA/CDNA wavefronts are 32-wide; only GFX8/GFX9 (not a Crane target) use
// 64-wide wavefronts, kept for fidelity with upstream's condition.
static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
#if defined(GGML_USE_HIP) && (defined(__GFX9__) || defined(__GFX8__))
    return 64;
#else
    return 32;
#endif
}

#define GGML_CUDA_CC_VOLTA  700
#define GGML_CUDA_CC_TURING 750
#define GGML_CUDA_CC_AMPERE 800

// AMD WMMA (RDNA3/RDNA4 tensor cores) and CDNA MFMA. Crane does not target
// CDNA, so `AMD_MFMA_AVAILABLE` is never defined here (`CDNA` is never
// defined above); the vendored CDNA code paths simply stay dead.
#if defined(GGML_USE_HIP) && (defined(RDNA4) || defined(RDNA3))
#define AMD_WMMA_AVAILABLE
#endif // defined(GGML_USE_HIP) && (defined(RDNA4) || defined(RDNA3))

// Volta instructions are unreachable from Crane's device query logic
// (Ampere+ only), but vendored code still parses Volta-guarded branches, so
// the macro is kept for the vendored files to compile unmodified.
#if !defined(GGML_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
#define VOLTA_MMA_AVAILABLE
#endif

#if !defined(GGML_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
#define TURING_MMA_AVAILABLE
#endif

#if !defined(GGML_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define AMPERE_MMA_AVAILABLE
#endif

#if !defined(GGML_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define CP_ASYNC_AVAILABLE
#endif

#define FLASH_ATTN_AVAILABLE

// `v_dot2_f32_f16` is an RDNA2+/CDNA-family instruction (and gfx906); Crane
// targets RDNA3+, so this is unconditionally true under HIP here.
#if defined(GGML_USE_HIP)
#define V_DOT2_F32_F16_AVAILABLE
#endif // defined(GGML_USE_HIP)

// Programmatic Dependent Launch (Blackwell-specific scheduling hint):
// no-ops, Crane does not target Blackwell in Phase 1. Both are called from
// device/kernel code (not host dispatch, which Crane's Rust layer replaces
// entirely), so both need `__device__`, not `__host__`.
static __device__ __forceinline__ void ggml_cuda_pdl_sync() {}
static __device__ __forceinline__ void ggml_cuda_pdl_lc() {}

#define GGML_CUDA_RESTRICT __restrict__

#define GGML_UNUSED(x) (void) (x)
#define GGML_UNUSED_VARS(...) \
    do { \
        (void) sizeof((__VA_ARGS__, 0)); \
    } while (0)

// hipcc defines `__CUDA_ARCH__` too, to a fixed compatibility value (an
// NVIDIA cc constant, e.g. 890) unrelated to the real AMD target, purely so
// CUDA-arch-gated headers parse under HIP. Confirmed empirically: a
// gfx1201 (RDNA4) build showed `__CUDA_ARCH__=890` in hipcc's own
// invocation. So `GGML_USE_HIP` must be checked *first*, never inferred
// from the absence of `__CUDA_ARCH__`, exactly like every
// `*_MMA_AVAILABLE` macro above already does. `__trap()` is a CUDA/nvcc-only
// intrinsic; hipcc (clang-based) has no such identifier and needs the
// portable `__builtin_trap()` instead (confirmed: nvcc accepts `__trap()`
// only, not `__builtin_trap()`, so this must stay backend-conditional
// rather than picking one spelling for both).
#if defined(GGML_USE_HIP)
#define NO_DEVICE_CODE __builtin_trap()
#elif defined(__CUDA_ARCH__)
#define NO_DEVICE_CODE __trap()
#else
#define NO_DEVICE_CODE
#endif

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1. `fastdiv_values`
// packs <mp, L, divisor> in <x, y, z> (see
// `ops::fused_ops::fattn::init_fastdiv_values`, the Rust-side producer).
static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
    const uint32_t hi = __umulhi(n, fastdiv_values.x);
    return (hi + n) >> fastdiv_values.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fastdiv_values) {
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z;
}

static __device__ __forceinline__ uint2 fast_div_modulo(uint32_t n, const uint3 fastdiv_values) {
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    const uint32_t mod_val = n - div_val * fastdiv_values.z;
    return make_uint2(div_val, mod_val);
}

/// ALiBi per-head slope. `max_bias <= 0` (Phase 1 always passes `max_bias =
/// 0`) short-circuits to `1.0`, so the `h`/`n_head_log2`/`m0`/`m1` ALiBi
/// bookkeeping is dead code until a future phase wires up a model that
/// needs it.
static __device__ __forceinline__ float get_alibi_slope(
    const float max_bias, const uint32_t h, const uint32_t n_head_log2, const float m0, const float m1) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int exph = h < n_head_log2 ? h + 1 : 2 * (h - n_head_log2) + 1;
    return powf(base, exph);
}

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const float v, const float u) {
    acc += v * u;
}

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const float2 v, const float2 u) {
    acc += v.x * u.x;
    acc += v.y * u.y;
}

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const half2 v, const half2 u) {
#ifdef V_DOT2_F32_F16_AVAILABLE
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(v), "v"(u));
#else
    const float2 tmpv = __half22float2(v);
    const float2 tmpu = __half22float2(u);
    acc += tmpv.x * tmpu.x;
    acc += tmpv.y * tmpu.y;
#endif // V_DOT2_F32_F16_AVAILABLE
}

// Maximum number of bytes that can be copied in a single instruction.
static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
#ifdef GGML_USE_HIP
    return 16;
#else
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    return 16;
#else
    return 8;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
#endif // GGML_USE_HIP
}

// Important: do not use this function if dst and src both point at
// registers. Due to the strict aliasing rule the compiler can do incorrect
// optimizations if src and dst have different types; this is intended for
// copies between registers and SRAM/VRAM, where dst/src are guaranteed not
// to alias.
template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    static_assert(
        nbytes <= ggml_cuda_get_max_cpy_bytes() || alignment == 0,
        "misusing the alignment parameter: only use it to work around an unaligned pointer, not to do more bytes "
        "per copy than ggml_cuda_get_max_cpy_bytes() allows (call this function in a loop instead)");
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;
#pragma unroll
    for (int i = 0; i < nbytes / nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2 *) dst)[i] = ((const int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}

// The compiler is always able to unroll loops if they contain continue
// expressions. In such cases loop unrolling can still be achieved via
// recursion.
template <int n>
struct ggml_cuda_unroll {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(n - 1, args...);
        ggml_cuda_unroll<n - 1>{}(f, args...);
    }
};

template <>
struct ggml_cuda_unroll<1> {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(0, args...);
    }
};

template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
    }
    return x;
}
