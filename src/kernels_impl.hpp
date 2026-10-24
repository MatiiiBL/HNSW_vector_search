#pragma once

// Internal: one kernel per (metric, instruction set). Each matches vecdex::kernels::DistanceFn.
// Only the sets that can exist on the target architecture are declared, and the
// VECDEX_HAVE_* macros tell kernels.cpp which ones to register.

#include <cstddef>

namespace vecdex::kernels::detail {

float l2_squared_scalar(const float* a, const float* b, std::size_t dim) noexcept;
float inner_product_scalar(const float* a, const float* b, std::size_t dim) noexcept;

// NEON is part of the baseline of every AArch64 CPU (Apple Silicon included): no runtime check.
#if defined(__aarch64__) && defined(__ARM_NEON)
#define VECDEX_HAVE_NEON 1
float l2_squared_neon(const float* a, const float* b, std::size_t dim) noexcept;
float inner_product_neon(const float* a, const float* b, std::size_t dim) noexcept;
#endif

// AVX2 + FMA is optional on x86-64, so it is compiled in always and enabled only if the CPU
// reports it at runtime. Needs GCC or Clang for the target attribute and CPU detection.
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#define VECDEX_HAVE_AVX2 1
float l2_squared_avx2(const float* a, const float* b, std::size_t dim) noexcept;
float inner_product_avx2(const float* a, const float* b, std::size_t dim) noexcept;
bool cpu_supports_avx2_fma() noexcept;
#endif

}  // namespace vecdex::kernels::detail
