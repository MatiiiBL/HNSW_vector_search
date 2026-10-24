#pragma once

#include <cstdint>
#include <span>

namespace vecdex {

// How an index compares vectors. Every metric is expressed as a distance: smaller is closer.
enum class Metric : std::uint8_t {
    L2,            // ||a - b||^2. Squared: same ranking as the true distance, without the sqrt.
    InnerProduct,  // 1 - <a, b>, for maximum inner product search (the hnswlib convention).
    Cosine,        // 1 - cos(a, b). Indexes normalize vectors to unit length when they store
                   // or query them, so this is computed as 1 - <a, b> on unit vectors.
};

// These use the fastest kernels this CPU supports (see kernels.hpp).
// Precondition: a.size() == b.size(), checked by assert only; indexes validate dimensions at
// their API boundary so the hot path doesn't have to.
float l2_squared(std::span<const float> a, std::span<const float> b) noexcept;
float inner_product(std::span<const float> a, std::span<const float> b) noexcept;

// The distance between a and b under `metric`. For Cosine, a and b must already have unit
// length.
float distance(Metric metric, std::span<const float> a, std::span<const float> b) noexcept;

}  // namespace vecdex
