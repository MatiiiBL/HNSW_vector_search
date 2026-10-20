#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "vecdex/matrix.hpp"
#include "vecdex/neighbor.hpp"

namespace vecdex {

// Exact k-nearest-neighbor search by scanning every stored vector: O(n * dim) per query.
// It is the correctness baseline every approximate index is measured against.
//
// Errors: std::invalid_argument for bad arguments (dimension mismatch, k == 0, NaN or
// infinite values). A failed call leaves the index unchanged.
class BruteForceIndex {
public:
    // Throws std::invalid_argument if dim is 0.
    explicit BruteForceIndex(std::size_t dim);

    std::size_t dim() const noexcept { return vectors_.dim(); }
    std::size_t size() const noexcept { return vectors_.rows(); }

    // Stores one vector and returns its id.
    Id add(std::span<const float> vector);
    // Stores every row of `vectors`; they get consecutive ids starting at size().
    void add(const Matrix<float>& vectors);

    // The min(k, size()) stored vectors closest to `query`, closest first, ties by smaller id.
    std::vector<Neighbor> search(std::span<const float> query, std::size_t k) const;

private:
    void check_can_add(std::span<const float> values, std::size_t count) const;

    Matrix<float> vectors_;
};

}  // namespace vecdex
