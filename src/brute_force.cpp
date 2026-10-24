#include "vecdex/brute_force.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>

#include "vecdex/kernels.hpp"

namespace vecdex {
namespace {

bool all_finite(std::span<const float> values) {
    return std::ranges::all_of(values, [](float v) { return std::isfinite(v); });
}

float norm(std::span<const float> v) noexcept {
    return std::sqrt(inner_product(v, v));
}

// False for a zero vector, or one whose norm overflows float.
bool can_normalize(std::span<const float> v) noexcept {
    const float n = norm(v);
    return n > 0.0F && std::isfinite(n);
}

// Precondition: can_normalize(v).
void normalize(std::span<float> v) noexcept {
    const float n = norm(v);
    for (float& x : v) {
        x /= n;
    }
}

// The k smallest of distance_of(0), ..., distance_of(n - 1) as Neighbors, closest first, ties
// by smaller id. A max-heap (under `closer`) holds the best candidates seen so far; its front
// is the worst of them, so rejecting a candidate costs one comparison and accepting one costs
// O(log k): O(n log k) in total with O(k) extra memory.
template <typename DistanceOf>
std::vector<Neighbor> select_top_k(std::size_t n, std::size_t k, DistanceOf distance_of) {
    const std::size_t keep = std::min(k, n);
    std::vector<Neighbor> heap;
    heap.reserve(keep);
    for (std::size_t i = 0; i < n; ++i) {
        const Neighbor candidate{.distance = distance_of(i), .id = static_cast<Id>(i)};
        if (heap.size() < keep) {
            heap.push_back(candidate);
            std::ranges::push_heap(heap, closer);
        } else if (closer(candidate, heap.front())) {
            std::ranges::pop_heap(heap, closer);  // moves the current worst to the back
            heap.back() = candidate;
            std::ranges::push_heap(heap, closer);
        }
    }
    std::ranges::sort_heap(heap, closer);  // ascending under `closer`: closest first
    return heap;
}

}  // namespace

BruteForceIndex::BruteForceIndex(std::size_t dim, Metric metric) : metric_(metric), vectors_(dim) {}

Id BruteForceIndex::add(std::span<const float> vector) {
    if (vector.size() != dim()) {
        throw std::invalid_argument(std::format(
            "BruteForceIndex::add: vector has dimension {}, index has {}", vector.size(), dim()));
    }
    check_can_add(vector, 1);
    const auto id = static_cast<Id>(size());
    store(vector);
    return id;
}

void BruteForceIndex::add(const Matrix<float>& vectors) {
    if (vectors.dim() != dim()) {
        throw std::invalid_argument(std::format(
            "BruteForceIndex::add: vectors have dimension {}, index has {}", vectors.dim(), dim()));
    }
    check_can_add(vectors.data(), vectors.rows());
    store(vectors.data());
}

// Every check runs before anything is stored, so a rejected batch leaves the index unchanged.
void BruteForceIndex::check_can_add(std::span<const float> values, std::size_t count) const {
    if (count > std::numeric_limits<Id>::max() - size()) {
        throw std::length_error("BruteForceIndex::add: too many vectors for 32-bit ids");
    }
    if (!all_finite(values)) {
        throw std::invalid_argument("BruteForceIndex::add: vectors must not contain NaN or inf");
    }
    if (metric_ == Metric::Cosine) {
        for (std::size_t r = 0; r < count; ++r) {
            if (!can_normalize(values.subspan(r * dim(), dim()))) {
                throw std::invalid_argument(
                    std::format("BruteForceIndex::add: vector {} has zero length, which the "
                                "cosine metric can't normalize",
                                r));
            }
        }
    }
}

// Appends already-validated rows. Normalizing can't fail, so the only possible exception is
// std::bad_alloc from append, which leaves the matrix unchanged.
void BruteForceIndex::store(std::span<const float> values) {
    const std::size_t first_new = size();
    vectors_.append(values);
    if (metric_ == Metric::Cosine) {
        for (std::size_t r = first_new; r < size(); ++r) {
            normalize(vectors_.row(r));
        }
    }
}

std::vector<Neighbor> BruteForceIndex::search(std::span<const float> query, std::size_t k) const {
    if (query.size() != dim()) {
        throw std::invalid_argument(std::format(
            "BruteForceIndex::search: query has dimension {}, index has {}", query.size(), dim()));
    }
    if (k == 0) {
        throw std::invalid_argument("BruteForceIndex::search: k must be positive");
    }
    if (!all_finite(query)) {
        throw std::invalid_argument("BruteForceIndex::search: query must not contain NaN or inf");
    }

    std::vector<float> unit_query;
    const float* q = query.data();
    if (metric_ == Metric::Cosine) {
        if (!can_normalize(query)) {
            throw std::invalid_argument(
                "BruteForceIndex::search: the cosine metric can't normalize a zero query");
        }
        unit_query.assign(query.begin(), query.end());
        normalize(unit_query);
        q = unit_query.data();
    }

    // The kernel is looked up once per query, and the metric branch is taken once, outside
    // the scan: the loop body is one indirect call plus the heap check.
    const std::size_t d = dim();
    if (metric_ == Metric::L2) {
        const kernels::DistanceFn l2 = kernels::active().l2_squared;
        return select_top_k(size(), k,
                            [&](std::size_t i) { return l2(q, vectors_.row(i).data(), d); });
    }
    const kernels::DistanceFn dot = kernels::active().inner_product;
    return select_top_k(size(), k,
                        [&](std::size_t i) { return 1.0F - dot(q, vectors_.row(i).data(), d); });
}

}  // namespace vecdex
