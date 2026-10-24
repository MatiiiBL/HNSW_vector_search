#include "vector_ops.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>

#include "vecdex/neighbor.hpp"

namespace vecdex::detail {
namespace {

float norm(std::span<const float> v) noexcept {
    return std::sqrt(inner_product(v, v));
}

}  // namespace

bool all_finite(std::span<const float> values) noexcept {
    return std::ranges::all_of(values, [](float v) { return std::isfinite(v); });
}

bool can_normalize(std::span<const float> v) noexcept {
    const float n = norm(v);
    return n > 0.0F && std::isfinite(n);
}

void normalize(std::span<float> v) noexcept {
    const float n = norm(v);
    for (float& x : v) {
        x /= n;
    }
}

void check_new_vectors(std::string_view where, const IndexShape& index, std::size_t given_dim,
                       std::span<const float> values) {
    if (given_dim != index.dim) {
        throw std::invalid_argument(
            std::format("{}: got dimension {}, index has {}", where, given_dim, index.dim));
    }
    const std::size_t count = values.size() / index.dim;
    if (count > std::numeric_limits<Id>::max() - index.size) {
        throw std::length_error(std::format("{}: too many vectors for 32-bit ids", where));
    }
    if (!all_finite(values)) {
        throw std::invalid_argument(std::format("{}: vectors must not contain NaN or inf", where));
    }
    if (index.metric == Metric::Cosine) {
        for (std::size_t r = 0; r < count; ++r) {
            if (!can_normalize(values.subspan(r * index.dim, index.dim))) {
                throw std::invalid_argument(std::format(
                    "{}: vector {} has zero length, which the cosine metric can't normalize", where,
                    r));
            }
        }
    }
}

void check_query(std::string_view where, const IndexShape& index, std::span<const float> query,
                 std::size_t k) {
    if (query.size() != index.dim) {
        throw std::invalid_argument(std::format("{}: query has dimension {}, index has {}", where,
                                                query.size(), index.dim));
    }
    if (k == 0) {
        throw std::invalid_argument(std::format("{}: k must be positive", where));
    }
    if (!all_finite(query)) {
        throw std::invalid_argument(std::format("{}: query must not contain NaN or inf", where));
    }
    if (index.metric == Metric::Cosine && !can_normalize(query)) {
        throw std::invalid_argument(
            std::format("{}: the cosine metric can't normalize a zero query", where));
    }
}

std::span<const float> prepare_query(Metric metric, std::span<const float> query,
                                     std::vector<float>& scratch) {
    if (metric != Metric::Cosine) {
        return query;
    }
    scratch.assign(query.begin(), query.end());
    normalize(scratch);
    return scratch;
}

}  // namespace vecdex::detail
