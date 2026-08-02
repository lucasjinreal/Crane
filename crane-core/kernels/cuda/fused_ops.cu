/**
 * Fused CUDA kernels for Crane transformer inference.
 *
 * Targets: sm_80+ (Ampere & newer, bf16 support), and AMD GPUs via HIP — the
 * ROCm launcher (`ops/fused_ops/rocm_impl.rs`) hands this same source to hipcc
 * at runtime through candle's shim, which shadows <cuda_bf16.h>/<cuda_fp16.h>.
 * The shim's __shfl_*_sync macros take an explicit `width`, so every shuffle
 * below passes WARP_SIZE: HIP honours it as a sub-wavefront shuffle, which is
 * what keeps the 32-lane block-reduction indexing correct on wave64 parts too.
 *
 * Kernels:
 *   1. fused_rmsnorm_residual_bf16  — RMSNorm + residual save
 *   2. fused_silu_mul_bf16          — SiLU(gate) * up  (one pass)
 *   3. fused_add_rmsnorm_bf16       — residual_add + RMSNorm (one pass)
 *   4. gpu_argmax_bf16              — GPU-side argmax over vocab (greedy decode)
 *   5. fused_residual_add_bf16      — residual += hidden, in place
 * Top-k lives in its own translation unit (`topk.cu`), sharing nothing here.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

// =====================================================================
// Helpers
// =====================================================================

static constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset, WARP_SIZE);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max_f32(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset, WARP_SIZE));
    }
    return val;
}

// Fast SiLU: x * sigmoid(x) = x / (1 + exp(-x))
__device__ __forceinline__ float fast_silu(float x) {
    return x / (1.0f + expf(-x));
}

// =====================================================================
// 1. Fused RMSNorm — one block per row
//
//    dst[row, col] = (x[row, col] / rms) * weight[col]
//    where rms = sqrt(mean(x²) + eps)
//
//    Identical to candle's rmsnorm but with explicit bf16 I/O and
//    handles up to 16384 columns per warp-tree reduction.
// =====================================================================

extern "C" __global__ void fused_rmsnorm_bf16(
    const __nv_bfloat16 *__restrict__ x,      // [rows, cols]
    __nv_bfloat16       *__restrict__ dst,     // [rows, cols]
    const __nv_bfloat16 *__restrict__ weight,  // [cols]
    const int ncols,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Phase 1: compute sum of squares
    float sum_sq = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        float v = __bfloat162float(x[row * ncols + col]);
        sum_sq += v * v;
    }

    // Warp reduce
    sum_sq = warp_reduce_sum_f32(sum_sq);

    // Cross-warp reduce via shared memory
    __shared__ float s_partial[32];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;

    if (lane_id == 0) s_partial[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum_f32(sum_sq);
        if (lane_id == 0) s_partial[0] = sum_sq;
    }
    __syncthreads();

    float scale = rsqrtf(s_partial[0] / (float)ncols + eps);

    // Phase 2: normalize and write output
    for (int col = tid; col < ncols; col += block_size) {
        float v = __bfloat162float(x[row * ncols + col]);
        float w = __bfloat162float(weight[col]);
        dst[row * ncols + col] = __float2bfloat16(v * scale * w);
    }
}

// =====================================================================
// 2. Fused SiLU(gate) * up — one pass over 2 * intermediate_size
//
//    Input:  gate_up [rows, 2*intermediate_size]  (gate||up concatenated)
//    Output: dst     [rows, intermediate_size]
//    dst[i] = silu(gate_up[i]) * gate_up[i + intermediate_size]
//
//    Saves 2 kernel launches (separate silu + mul) and 1 intermediate
//    tensor allocation.
// =====================================================================

extern "C" __global__ void fused_silu_mul_bf16(
    const __nv_bfloat16 *__restrict__ gate_up,  // [rows, 2*intermediate_size]
    __nv_bfloat16       *__restrict__ dst,       // [rows, intermediate_size]
    const int intermediate_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const __nv_bfloat16 *gate_row = gate_up + row * 2 * intermediate_size;
    const __nv_bfloat16 *up_row   = gate_row + intermediate_size;
    __nv_bfloat16       *dst_row  = dst + row * intermediate_size;

    for (int i = tid; i < intermediate_size; i += block_size) {
        float g = __bfloat162float(gate_row[i]);
        float u = __bfloat162float(up_row[i]);
        dst_row[i] = __float2bfloat16(fast_silu(g) * u);
    }
}

// f16 variant
extern "C" __global__ void fused_silu_mul_f16(
    const __half *__restrict__ gate_up,
    __half       *__restrict__ dst,
    const int intermediate_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const __half *gate_row = gate_up + row * 2 * intermediate_size;
    const __half *up_row   = gate_row + intermediate_size;
    __half       *dst_row  = dst + row * intermediate_size;

    for (int i = tid; i < intermediate_size; i += block_size) {
        float g = __half2float(gate_row[i]);
        float u = __half2float(up_row[i]);
        dst_row[i] = __float2half(fast_silu(g) * u);
    }
}

// f32 variant
extern "C" __global__ void fused_silu_mul_f32(
    const float *__restrict__ gate_up,
    float       *__restrict__ dst,
    const int intermediate_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const float *gate_row = gate_up + row * 2 * intermediate_size;
    const float *up_row   = gate_row + intermediate_size;
    float       *dst_row  = dst + row * intermediate_size;

    for (int i = tid; i < intermediate_size; i += block_size) {
        float g = gate_row[i];
        float u = up_row[i];
        dst_row[i] = fast_silu(g) * u;
    }
}

// =====================================================================
// 3. Fused residual_add + RMSNorm — one read of hidden, write norm + residual
//
//    residual[row] += hidden[row]            (in-place update)
//    dst[row] = rmsnorm(residual[row]) * weight
//
//    Eliminates the separate add kernel + RMSNorm kernel + extra read.
// =====================================================================

extern "C" __global__ void fused_add_rmsnorm_bf16(
    __nv_bfloat16       *__restrict__ residual,  // [rows, cols] — updated in-place
    const __nv_bfloat16 *__restrict__ hidden,    // [rows, cols] — value to add
    __nv_bfloat16       *__restrict__ dst,        // [rows, cols] — normalized output
    const __nv_bfloat16 *__restrict__ weight,     // [cols]
    const int ncols,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int row_offset = row * ncols;

    // Phase 1: add residual, compute sum of squares
    float sum_sq = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        float r = __bfloat162float(residual[row_offset + col]);
        float h = __bfloat162float(hidden[row_offset + col]);
        float v = r + h;
        // Write residual back (in-place update)
        residual[row_offset + col] = __float2bfloat16(v);
        sum_sq += v * v;
    }

    // Warp + cross-warp reduce
    sum_sq = warp_reduce_sum_f32(sum_sq);
    __shared__ float s_partial[32];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;

    if (lane_id == 0) s_partial[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum_f32(sum_sq);
        if (lane_id == 0) s_partial[0] = sum_sq;
    }
    __syncthreads();

    float scale = rsqrtf(s_partial[0] / (float)ncols + eps);

    // Phase 2: normalize from updated residual
    for (int col = tid; col < ncols; col += block_size) {
        float v = __bfloat162float(residual[row_offset + col]);
        float w = __bfloat162float(weight[col]);
        dst[row_offset + col] = __float2bfloat16(v * scale * w);
    }
}

// =====================================================================
// 4. GPU Argmax — two-phase reduction for vocab-size vectors
//
//    Phase 1: Each block reduces a chunk of rows → per-block max + argmax
//    Phase 2: Single block reduces per-block results → final argmax
//
//    For greedy decode: replaces 303KB DtoH + CPU argmax with
//    a single scalar (4 bytes) DtoH.
// =====================================================================

extern "C" __global__ void gpu_argmax_bf16_phase1(
    const __nv_bfloat16 *__restrict__ logits,  // [vocab_size]
    float               *__restrict__ block_max_vals,
    int32_t             *__restrict__ block_max_idxs,
    const int vocab_size
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int bid = blockIdx.x;
    const int num_blocks = gridDim.x;

    // Each block handles a strided chunk
    int chunk = (vocab_size + num_blocks - 1) / num_blocks;
    int start = bid * chunk;
    int end   = min(start + chunk, vocab_size);

    float local_max = -INFINITY;
    int   local_idx = -1;

    for (int i = start + tid; i < end; i += block_size) {
        float v = __bfloat162float(logits[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    // Warp reduce
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, local_max, offset, WARP_SIZE);
        int   other_idx = __shfl_down_sync(0xffffffff, local_idx, offset, WARP_SIZE);
        if (other_val > local_max) {
            local_max = other_val;
            local_idx = other_idx;
        }
    }

    // Cross-warp reduce
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;

    __shared__ float s_max_vals[32];
    __shared__ int   s_max_idxs[32];

    if (lane_id == 0) {
        s_max_vals[warp_id] = local_max;
        s_max_idxs[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < num_warps) {
        local_max = s_max_vals[lane_id];
        local_idx = s_max_idxs[lane_id];

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, local_max, offset, WARP_SIZE);
            int   other_idx = __shfl_down_sync(0xffffffff, local_idx, offset, WARP_SIZE);
            if (other_val > local_max) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            block_max_vals[bid] = local_max;
            block_max_idxs[bid] = local_idx;
        }
    }
}

extern "C" __global__ void gpu_argmax_phase2(
    const float   *__restrict__ block_max_vals,
    const int32_t *__restrict__ block_max_idxs,
    int32_t       *__restrict__ output_token,
    const int num_blocks
) {
    const int tid = threadIdx.x;

    float best_val = -INFINITY;
    int   best_idx = -1;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float v = block_max_vals[i];
        if (v > best_val) {
            best_val = v;
            best_idx = block_max_idxs[i];
        }
    }

    // Warp reduce
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, best_val, offset, WARP_SIZE);
        int   other_idx = __shfl_down_sync(0xffffffff, best_idx, offset, WARP_SIZE);
        if (other_val > best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    // Cross-warp reduce
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    __shared__ float s_vals[32];
    __shared__ int   s_idxs[32];

    if (lane_id == 0) {
        s_vals[warp_id] = best_val;
        s_idxs[warp_id] = best_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x / WARP_SIZE;
        best_val = (lane_id < num_warps) ? s_vals[lane_id] : -INFINITY;
        best_idx = (lane_id < num_warps) ? s_idxs[lane_id] : -1;

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, best_val, offset, WARP_SIZE);
            int   other_idx = __shfl_down_sync(0xffffffff, best_idx, offset, WARP_SIZE);
            if (other_val > best_val) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            *output_token = best_idx;
        }
    }
}

// =====================================================================
// 5. Fused residual_add in-place: residual += hidden
//    Simple element-wise kernel to avoid candle's tensor add overhead.
// =====================================================================

extern "C" __global__ void fused_residual_add_bf16(
    __nv_bfloat16       *__restrict__ residual,  // [n] — updated in-place
    const __nv_bfloat16 *__restrict__ hidden,    // [n]
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float r = __bfloat162float(residual[idx]);
        float h = __bfloat162float(hidden[idx]);
        residual[idx] = __float2bfloat16(r + h);
    }
}
