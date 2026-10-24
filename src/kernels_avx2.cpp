#include "kernels_impl.hpp"

#ifdef VECDEX_HAVE_AVX2

#include <immintrin.h>

// This file is compiled WITHOUT -mavx2. Only the functions marked [[gnu::target("avx2,fma")]]
// may use AVX2 instructions, and they run only after cpu_supports_avx2_fma() said yes.
// Compiling the whole file with -mavx2 would be the ODR trap: any inline function it
// instantiates (from the standard library, say) could be emitted with AVX2 instructions, and
// the linker may keep that copy for the whole program, which then crashes on CPUs without AVX2.

namespace vecdex::kernels::detail {
namespace {

// Adds the 8 lanes of v.
[[gnu::target("avx2,fma")]] float horizontal_sum(__m256 v) noexcept {
    const __m128 halves = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 pairs = _mm_add_ps(halves, _mm_movehl_ps(halves, halves));
    const __m128 total = _mm_add_ss(pairs, _mm_movehdup_ps(pairs));
    return _mm_cvtss_f32(total);
}

}  // namespace

// Same structure as the NEON kernels, at 8 floats per register: 32 floats per iteration in
// four independent accumulators, then 8 at a time, then a scalar tail. Loads are unaligned
// (_mm256_loadu_ps), so callers need no particular alignment.

[[gnu::target("avx2,fma")]] float l2_squared_avx2(const float* a, const float* b,
                                                  std::size_t dim) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; dim - i >= 32; i += 32) {
        const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        const __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        const __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
    }
    for (; dim - i >= 8; i += 8) {
        const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc0 = _mm256_fmadd_ps(d, d, acc0);
    }
    float sum = horizontal_sum(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
    for (; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

[[gnu::target("avx2,fma")]] float inner_product_avx2(const float* a, const float* b,
                                                     std::size_t dim) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; dim - i >= 32; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }
    for (; dim - i >= 8; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }
    float sum = horizontal_sum(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Checks both the CPU and the OS: the compiler's runtime only reports AVX2 if the OS also
// saves the 256-bit registers on context switches.
bool cpu_supports_avx2_fma() noexcept {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

}  // namespace vecdex::kernels::detail

#endif  // VECDEX_HAVE_AVX2
