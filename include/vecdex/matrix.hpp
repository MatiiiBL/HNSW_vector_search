#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace vecdex {

// A dense rows x dim matrix stored contiguously in row-major order: row i occupies
// data()[i * dim() .. (i + 1) * dim()). Holds vectors (float) and ground-truth ids (int32).
//
// Spans returned by row() and data() point into the matrix's buffer. append() and
// reserve_rows() may reallocate that buffer, which invalidates every span taken earlier.
//
// rows() is derived from the buffer size instead of being stored, so the implicit copy and
// move operations are correct (rule of zero): a moved-from Matrix is simply empty.
template <typename T>
class Matrix {
public:
    // An empty matrix whose rows will have `dim` elements. Throws if dim is 0.
    explicit Matrix(std::size_t dim) : Matrix(0, dim) {}

    // A rows x dim matrix with every element value-initialized (zero for numbers).
    Matrix(std::size_t rows, std::size_t dim) : dim_(dim) {
        if (dim == 0) {
            throw std::invalid_argument("Matrix: dim must be positive");
        }
        data_.resize(checked_size(rows));
    }

    std::size_t rows() const noexcept { return data_.size() / dim_; }
    std::size_t dim() const noexcept { return dim_; }
    bool empty() const noexcept { return data_.empty(); }

    // Precondition: i < rows(). Checked by assert only, like std::vector::operator[].
    std::span<T> row(std::size_t i) noexcept {
        assert(i < rows());
        return {data_.data() + i * dim_, dim_};
    }
    std::span<const T> row(std::size_t i) const noexcept {
        assert(i < rows());
        return {data_.data() + i * dim_, dim_};
    }

    // All elements, row after row.
    std::span<T> data() noexcept { return data_; }
    std::span<const T> data() const noexcept { return data_; }

    void reserve_rows(std::size_t rows) { data_.reserve(checked_size(rows)); }

    // Appends values.size() / dim() rows, given back to back in row-major order.
    // values may point into this matrix (m.append(m.row(0)) is fine).
    // Throws if values.size() is not a multiple of dim(); the matrix is then unchanged.
    void append(std::span<const T> values) {
        if (values.size() % dim_ != 0) {
            throw std::invalid_argument("Matrix::append: size is not a multiple of dim");
        }
        if (aliases_storage(values)) {
            // vector::insert with a range from the same vector is undefined behavior: growing
            // would free the buffer `values` points into. Copy by offset after resizing.
            const auto offset = static_cast<std::size_t>(values.data() - data_.data());
            const std::size_t old_size = data_.size();
            data_.resize(old_size + values.size());
            std::copy_n(data_.data() + offset, values.size(), data_.data() + old_size);
        } else {
            data_.insert(data_.end(), values.begin(), values.end());
        }
    }

private:
    // rows * dim_, refusing to wrap around: a wrapped product would allocate a tiny buffer
    // that every later row() access overruns.
    std::size_t checked_size(std::size_t rows) const {
        if (rows > std::numeric_limits<std::size_t>::max() / dim_) {
            throw std::length_error("Matrix: rows * dim overflows size_t");
        }
        return rows * dim_;
    }

    bool aliases_storage(std::span<const T> values) const noexcept {
        // std::less gives a total order even for pointers into unrelated objects.
        const T* begin = data_.data();
        const T* end = begin + data_.size();
        return !values.empty() && !std::less<const T*>{}(values.data(), begin) &&
               std::less<const T*>{}(values.data(), end);
    }

    std::size_t dim_;
    std::vector<T> data_;
};

}  // namespace vecdex
