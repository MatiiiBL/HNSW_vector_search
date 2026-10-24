#include "vecdex/brute_force.hpp"

#include <algorithm>

#include "vecdex/kernels.hpp"
#include "vector_ops.hpp"

namespace vecdex {
namespace {

detail::IndexShape shape_of(const BruteForceIndex& index) {
    return {.metric = index.metric(), .dim = index.dim(), .size = index.size()};
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
    detail::check_new_vectors("BruteForceIndex::add", shape_of(*this), vector.size(), vector);
    const auto id = static_cast<Id>(size());
    store(vector);
    return id;
}

void BruteForceIndex::add(const Matrix<float>& vectors) {
    detail::check_new_vectors("BruteForceIndex::add", shape_of(*this), vectors.dim(),
                              vectors.data());
    store(vectors.data());
}

// Appends already-validated rows. Normalizing can't fail, so the only possible exception is
// std::bad_alloc from append, which leaves the matrix unchanged.
void BruteForceIndex::store(std::span<const float> values) {
    const std::size_t first_new = size();
    vectors_.append(values);
    if (metric_ == Metric::Cosine) {
        for (std::size_t r = first_new; r < size(); ++r) {
            detail::normalize(vectors_.row(r));
        }
    }
}

std::vector<Neighbor> BruteForceIndex::search(std::span<const float> query, std::size_t k) const {
    detail::check_query("BruteForceIndex::search", shape_of(*this), query, k);
    std::vector<float> scratch;
    const float* q = detail::prepare_query(metric_, query, scratch).data();

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
