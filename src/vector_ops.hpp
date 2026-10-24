#pragma once

// Internal: the validation, normalization and distance plumbing shared by every index, so
// BruteForceIndex and HnswIndex accept and reject exactly the same inputs.

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

#include "vecdex/distance.hpp"
#include "vecdex/kernels.hpp"

namespace vecdex::detail {

bool all_finite(std::span<const float> values) noexcept;

// False for a zero vector, or one whose length overflows float.
bool can_normalize(std::span<const float> v) noexcept;

// Scales v to unit length. Precondition: can_normalize(v).
void normalize(std::span<float> v) noexcept;

// What the shared checks need to know about an index.
struct IndexShape {
    Metric metric;
    std::size_t dim;
    std::size_t size;
};

// The checks every index runs before storing vectors given back to back in `values`, each of
// length `given_dim`. They run before anything is stored, so a rejected call changes nothing.
// `where` starts each error message, e.g. "HnswIndex::add".
void check_new_vectors(std::string_view where, const IndexShape& index, std::size_t given_dim,
                       std::span<const float> values);

// The checks every index runs on a query before searching.
void check_query(std::string_view where, const IndexShape& index, std::span<const float> query,
                 std::size_t k);

// The query as the index compares it: for Cosine a normalized copy (kept in `scratch`),
// otherwise the query itself.
std::span<const float> prepare_query(Metric metric, std::span<const float> query,
                                     std::vector<float>& scratch);

// The ranking distance under a metric (smaller is closer), bound once to the active kernel.
class MetricDistance {
public:
    MetricDistance(Metric metric, std::size_t dim) noexcept
        : kernel_(metric == Metric::L2 ? kernels::active().l2_squared
                                       : kernels::active().inner_product),
          one_minus_(metric != Metric::L2),
          dim_(dim) {}

    float operator()(const float* a, const float* b) const noexcept {
        const float raw = kernel_(a, b, dim_);
        return one_minus_ ? 1.0F - raw : raw;
    }

private:
    kernels::DistanceFn kernel_;
    bool one_minus_;  // inner product and cosine rank by 1 - <a, b>
    std::size_t dim_;
};

}  // namespace vecdex::detail
