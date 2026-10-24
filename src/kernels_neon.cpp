#include "kernels_impl.hpp"

#ifdef VECDEX_HAVE_NEON

#include <arm_neon.h>

namespace vecdex::kernels::detail {

// Both kernels process 16 floats per iteration in four independent accumulators. An FMA
// can't start until the previous FMA into the same register finishes (several cycles), so a
// single accumulator would leave the floating-point units idle most of the time.
// Loads are unaligned (vld1q_f32 has no alignment requirement), and the last dim % 4
// elements go through a scalar tail, so nothing is ever read past the end of the inputs.

float l2_squared_neon(const float* a, const float* b, std::size_t dim) noexcept {
    float32x4_t acc0 = vdupq_n_f32(0.0F);
    float32x4_t acc1 = vdupq_n_f32(0.0F);
    float32x4_t acc2 = vdupq_n_f32(0.0F);
    float32x4_t acc3 = vdupq_n_f32(0.0F);
    std::size_t i = 0;
    for (; dim - i >= 16; i += 16) {
        const float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        const float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        const float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        const float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
        acc2 = vfmaq_f32(acc2, d2, d2);
        acc3 = vfmaq_f32(acc3, d3, d3);
    }
    for (; dim - i >= 4; i += 4) {
        const float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        acc0 = vfmaq_f32(acc0, d, d);
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float inner_product_neon(const float* a, const float* b, std::size_t dim) noexcept {
    float32x4_t acc0 = vdupq_n_f32(0.0F);
    float32x4_t acc1 = vdupq_n_f32(0.0F);
    float32x4_t acc2 = vdupq_n_f32(0.0F);
    float32x4_t acc3 = vdupq_n_f32(0.0F);
    std::size_t i = 0;
    for (; dim - i >= 16; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; dim - i >= 4; i += 4) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

}  // namespace vecdex::kernels::detail

#endif  // VECDEX_HAVE_NEON
