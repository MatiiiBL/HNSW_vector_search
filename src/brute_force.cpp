#include "vecdex/brute_force.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>

#include "vecdex/distance.hpp"

namespace vecdex {
namespace {

bool all_finite(std::span<const float> values) {
    return std::ranges::all_of(values, [](float v) { return std::isfinite(v); });
}

}  // namespace

BruteForceIndex::BruteForceIndex(std::size_t dim) : vectors_(dim) {}

Id BruteForceIndex::add(std::span<const float> vector) {
    if (vector.size() != dim()) {
        throw std::invalid_argument(std::format(
            "BruteForceIndex::add: vector has dimension {}, index has {}", vector.size(), dim()));
    }
    check_can_add(vector, 1);
    const auto id = static_cast<Id>(size());
    vectors_.append(vector);
    return id;
}

void BruteForceIndex::add(const Matrix<float>& vectors) {
    if (vectors.dim() != dim()) {
        throw std::invalid_argument(std::format(
            "BruteForceIndex::add: vectors have dimension {}, index has {}", vectors.dim(), dim()));
    }
    check_can_add(vectors.data(), vectors.rows());
    vectors_.append(vectors.data());
}

// Every check runs before anything is stored, so a rejected batch leaves the index unchanged.
void BruteForceIndex::check_can_add(std::span<const float> values, std::size_t count) const {
    if (count > std::numeric_limits<Id>::max() - size()) {
        throw std::length_error("BruteForceIndex::add: too many vectors for 32-bit ids");
    }
    if (!all_finite(values)) {
        throw std::invalid_argument("BruteForceIndex::add: vectors must not contain NaN or inf");
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

    // A max-heap (under `closer`) of the best candidates seen so far. Its front is the worst
    // of them, so rejecting a candidate costs one comparison and accepting one costs
    // O(log k): O(n log k) in total with O(k) extra memory.
    const std::size_t keep = std::min(k, size());
    std::vector<Neighbor> heap;
    heap.reserve(keep);
    for (std::size_t i = 0; i < size(); ++i) {
        const Neighbor candidate{.distance = l2_squared(query, vectors_.row(i)),
                                 .id = static_cast<Id>(i)};
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

}  // namespace vecdex
