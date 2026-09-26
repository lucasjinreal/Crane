// Fused elementwise kernels for the Intel SYCL backend — the counterpart of
// `kernels/cuda/fused_ops.cu`. Built into `libcrane_gdn_sycl.so` by
// `crane-core/build.rs` (icpx, `--features sycl` only) and driven by
// `ops/fused_ops/sycl_impl.rs`.
//
// `fused_silu_mul` collapses the `narrow(gate) + narrow(up) + silu + mul` op
// chain (candle: 2 real kernel launches after narrow's free views — SiLU then
// multiply) into one submission, mirroring the CUDA/ROCm `fused_silu_mul_*`
// kernels.
#include <sycl/sycl.hpp>

namespace {

// dtype tags match `ops/fused_ops/sycl_impl.rs`.
enum { CRANE_FSM_F32 = 0, CRANE_FSM_F16 = 1 };

template <typename T>
void fused_silu_mul_launch(sycl::queue &q, const T *gu, T *out,
                           long long n_rows, int isz) {
  const long long total = n_rows * static_cast<long long>(isz);
  q.parallel_for(sycl::range<1>(static_cast<size_t>(total)),
                 [=](sycl::id<1> idx) {
                   long long i = idx[0];
                   long long row = i / isz;
                   long long col = i % isz;
                   const T *base = gu + row * (2LL * isz);
                   float g = static_cast<float>(base[col]);
                   float u = static_cast<float>(base[isz + col]);
                   float silu = g / (1.0f + sycl::exp(-g));
                   out[i] = static_cast<T>(silu * u);
                 });
}

} // namespace

extern "C" int crane_fused_silu_mul_sycl(void *queue, int dtype,
                                        const void *gate_up, void *out,
                                        long long n_rows, int intermediate_size) {
  try {
    auto &sq = *static_cast<sycl::queue *>(queue);
    switch (dtype) {
    case CRANE_FSM_F32:
      fused_silu_mul_launch<float>(sq, static_cast<const float *>(gate_up),
                                   static_cast<float *>(out), n_rows,
                                   intermediate_size);
      return 0;
    case CRANE_FSM_F16:
      fused_silu_mul_launch<sycl::half>(
          sq, static_cast<const sycl::half *>(gate_up),
          static_cast<sycl::half *>(out), n_rows, intermediate_size);
      return 0;
    default:
      return 2; // unsupported dtype
    }
  } catch (const sycl::exception &) {
    return 1;
  } catch (...) {
    return 1;
  }
}
