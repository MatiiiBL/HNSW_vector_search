#pragma once

#include <cassert>
#include <cstddef>
#include <span>

namespace vecdex {

// Squared Euclidean distance. It ranks neighbors exactly like the true distance (sqrt is
// monotonic) and skips the sqrt. Precondition: a.size() == b.size(); callers validate
// dimensions at the API boundary so this hot loop doesn't have to.
//
// Plain scalar loop: the reference implementation the SIMD kernels are tested against (M2).
inline float l2_squared(std::span<const float> a, std::span<const float> b) noexcept {
    assert(a.size() == b.size());
    float sum = 0.0F;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

}  // namespace vecdex
