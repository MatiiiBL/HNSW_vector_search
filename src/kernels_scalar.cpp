#include "kernels_impl.hpp"

namespace vecdex::kernels::detail {

// The reference kernels: one accumulator, strictly left to right. Without -ffast-math the
// compiler may not reorder the additions (float addition isn't associative), so the
// reduction stays sequential: correct everywhere, and the baseline the SIMD kernels must beat.

float l2_squared_scalar(const float* a, const float* b, std::size_t dim) noexcept {
    float sum = 0.0F;
    for (std::size_t i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float inner_product_scalar(const float* a, const float* b, std::size_t dim) noexcept {
    float sum = 0.0F;
    for (std::size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

}  // namespace vecdex::kernels::detail
