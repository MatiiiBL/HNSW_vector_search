#include "vecdex/distance.hpp"

#include <cassert>
#include <limits>

#include "vecdex/kernels.hpp"

namespace vecdex {

float l2_squared(std::span<const float> a, std::span<const float> b) noexcept {
    assert(a.size() == b.size());
    return kernels::active().l2_squared(a.data(), b.data(), a.size());
}

float inner_product(std::span<const float> a, std::span<const float> b) noexcept {
    assert(a.size() == b.size());
    return kernels::active().inner_product(a.data(), b.data(), a.size());
}

float distance(Metric metric, std::span<const float> a, std::span<const float> b) noexcept {
    switch (metric) {
        case Metric::L2:
            return l2_squared(a, b);
        case Metric::InnerProduct:
        case Metric::Cosine:
            return 1.0F - inner_product(a, b);
    }
    assert(false && "unhandled Metric");
    return std::numeric_limits<float>::quiet_NaN();
}

}  // namespace vecdex
