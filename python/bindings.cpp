// The extension module vecdex._core: HnswIndex exposed to Python as vecdex.Index, with a NumPy
// API in the style of hnswlib. The vecdex package re-exports it; python/README.md is the user
// documentation.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vecdex/distance.hpp"
#include "vecdex/hnsw.hpp"
#include "vecdex/matrix.hpp"
#include "vecdex/neighbor.hpp"

namespace py = pybind11;

namespace {

using vecdex::HnswIndex;
using vecdex::Metric;

// knn_query's search breadth unless the caller sets one. Measured on SIFT1M, M=16,
// efConstruction=200: recall@10 is 0.9631 at ef = 64; at hnswlib's default of 10 it is 0.7060
// (0.7107 for hnswlib itself).
constexpr std::size_t kDefaultEf = 64;

// The metric names hnswlib uses.
Metric parse_space(std::string_view space) {
    if (space == "l2") {
        return Metric::L2;
    }
    if (space == "ip") {
        return Metric::InnerProduct;
    }
    if (space == "cosine") {
        return Metric::Cosine;
    }
    throw py::value_error(std::format("space must be 'l2', 'ip' or 'cosine', got '{}'", space));
}

std::string_view space_name(Metric metric) noexcept {
    switch (metric) {
        case Metric::L2:
            return "l2";
        case Metric::InnerProduct:
            return "ip";
        case Metric::Cosine:
            return "cosine";
    }
    return "unknown";
}

// Python ints are signed; the C++ API takes size_t. pybind11 has already refused values that
// don't fit in 64 bits; this refuses zero and negative ones.
std::size_t positive(std::int64_t value, std::string_view name) {
    if (value < 1) {
        throw py::value_error(std::format("{} must be positive, got {}", name, value));
    }
    return static_cast<std::size_t>(value);
}

// num_threads as the C++ API takes it, where 0 means one thread per hardware thread.
std::size_t thread_count(std::int64_t value) {
    if (value < 0) {
        throw py::value_error(std::format(
            "num_threads must be positive, or 0 for one per hardware thread, got {}", value));
    }
    return static_cast<std::size_t>(value);
}

// True if the buffer holds native-endian 4-byte floats. NumPy describes an aligned float32
// array as "f" but an unaligned one as "=f", so a native byte-order prefix is accepted here and
// alignment is checked separately.
bool holds_float32(const py::buffer_info& info) {
    constexpr char kNativeOrder = std::endian::native == std::endian::little ? '<' : '>';
    std::string_view format = info.format;
    if (!format.empty() &&
        (format.front() == '@' || format.front() == '=' || format.front() == kNativeOrder)) {
        format.remove_prefix(1);
    }
    return format == "f" && std::cmp_equal(info.itemsize, sizeof(float));
}

// C order as NumPy defines it: the stride of an axis of length 1 doesn't matter, and an empty
// array is contiguous.
bool is_c_contiguous(const py::buffer_info& info) {
    if (info.size == 0) {
        return true;
    }
    py::ssize_t expected = info.itemsize;
    for (std::size_t axis = info.shape.size(); axis-- > 0;) {
        if (info.shape[axis] != 1 && info.strides[axis] != expected) {
            return false;
        }
        expected *= info.shape[axis];
    }
    return true;
}

// For error messages: the NumPy dtype when there is one, otherwise the buffer's format code.
std::string item_type(const py::buffer& data, const py::buffer_info& info) {
    if (py::hasattr(data, "dtype")) {
        return py::str(data.attr("dtype")).cast<std::string>();
    }
    return std::format("buffer format '{}'", info.format);
}

// float32 vectors, one per row, read in place from a buffer-protocol object (in practice a NumPy
// array): a 2-D (n, dim) array, or a 1-D (dim,) array as one row. Anything else is rejected with
// an error that says how to convert it explicitly, because converting it here would silently
// copy the whole input.
//
// While a Rows exists it holds the buffer, which keeps the memory alive and in place, so the
// rows may be read with the GIL released. Create and destroy it with the GIL held.
class Rows {
public:
    Rows(const py::buffer& data, std::size_t dim, std::string_view method)
        : buffer_(data.request()), dim_(dim) {
        if (buffer_.ndim != 1 && buffer_.ndim != 2) {
            throw py::value_error(std::format(
                "{}: data must be 1-D (one vector) or 2-D (one vector per row), got {}-D", method,
                buffer_.ndim));
        }
        if (!holds_float32(buffer_)) {
            throw py::type_error(std::format(
                "{}: data must be float32, got {}; data.astype(np.float32) converts it (a copy)",
                method, item_type(data, buffer_)));
        }
        const auto given_dim = static_cast<std::size_t>(buffer_.shape.back());
        if (given_dim != dim) {
            throw py::value_error(std::format("{}: vectors have dimension {}, the index has {}",
                                              method, given_dim, dim));
        }
        if (!is_c_contiguous(buffer_)) {
            throw py::value_error(std::format(
                "{}: data must be C-contiguous; np.ascontiguousarray(data) makes a contiguous copy",
                method));
        }
        if (reinterpret_cast<std::uintptr_t>(buffer_.ptr) % alignof(float) != 0) {
            throw py::value_error(std::format(
                "{}: data is not aligned for float32; data.copy() makes an aligned copy", method));
        }
        count_ = buffer_.ndim == 2 ? static_cast<std::size_t>(buffer_.shape[0]) : 1;
    }

    std::size_t count() const noexcept { return count_; }

    // The caller's memory itself, not a copy.
    vecdex::MatrixView<float> view() const {
        return {std::span(static_cast<const float*>(buffer_.ptr), count_ * dim_), dim_};
    }

private:
    py::buffer_info buffer_;
    std::size_t dim_;
    std::size_t count_ = 0;
};

// An HnswIndex as Python sees it.
//
// With the GIL released, Python threads really do run these methods at the same time, so a
// reader-writer lock enforces HnswIndex's contract rather than trusting every caller to: any
// number of searches at once, an add only alone. Each method takes the lock after releasing the
// GIL and never needs the GIL while holding the lock, so the two can't deadlock. ef_ is state of
// the binding only; it is read and written with the GIL held.
class PyIndex {
public:
    PyIndex(std::string_view space, std::int64_t dim, std::int64_t m, std::int64_t ef_construction,
            std::uint64_t seed)
        : index_(positive(dim, "dim"), parse_space(space),
                 {.M = positive(m, "M"),
                  .ef_construction = positive(ef_construction, "ef_construction"),
                  .seed = seed}) {}

    // Fixed at construction, so they need no lock.
    std::size_t dim() const noexcept { return index_.dim(); }
    std::string_view space() const noexcept { return space_name(index_.metric()); }
    std::size_t m() const noexcept { return index_.params().M; }
    std::size_t ef_construction() const noexcept { return index_.params().ef_construction; }

    std::size_t ef() const noexcept { return ef_; }
    void set_ef(std::int64_t ef) { ef_ = positive(ef, "ef"); }

    std::size_t size() const {
        const py::gil_scoped_release release;
        const std::shared_lock lock(mutex_);
        return index_.size();
    }

    std::size_t memory_bytes() const {
        const py::gil_scoped_release release;
        const std::shared_lock lock(mutex_);
        return index_.memory_bytes();
    }

    py::array_t<std::int64_t> add_items(const py::buffer& data, std::int64_t num_threads) {
        const Rows rows(data, dim(), "add_items");
        const std::size_t threads = thread_count(num_threads);
        // Allocated before adding, so a failed allocation can't leave vectors added without ids.
        py::array_t<std::int64_t> ids(static_cast<py::ssize_t>(rows.count()));
        std::size_t first = 0;
        {
            const py::gil_scoped_release release;
            const std::unique_lock lock(mutex_);
            first = index_.size();
            // HnswIndex::add checks the whole batch before it stores any row, so a rejected call
            // adds nothing.
            index_.add(rows.view(), threads);
        }
        const std::span<std::int64_t> out(ids.mutable_data(), rows.count());
        std::iota(out.begin(), out.end(), static_cast<std::int64_t>(first));
        return ids;
    }

    std::pair<py::array_t<std::int64_t>, py::array_t<float>> knn_query(
        const py::buffer& data, std::int64_t k, std::optional<std::int64_t> ef,
        std::int64_t num_threads) const {
        const Rows queries(data, dim(), "knn_query");
        const std::size_t want = positive(k, "k");
        const std::size_t breadth = ef ? positive(*ef, "ef") : ef_;
        const std::size_t threads = thread_count(num_threads);
        // Results go straight into new NumPy arrays: one allocation each, no per-row objects.
        const std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(queries.count()),
                                             static_cast<py::ssize_t>(want)};
        py::array_t<std::int64_t> ids(shape);
        py::array_t<float> distances(shape);
        const std::span<std::int64_t> all_ids(ids.mutable_data(), queries.count() * want);
        const std::span<float> all_distances(distances.mutable_data(), queries.count() * want);
        const auto write_row = [&](std::size_t q, std::span<const vecdex::Neighbor> found) {
            const std::span<std::int64_t> row_ids = all_ids.subspan(q * want, want);
            const std::span<float> row_distances = all_distances.subspan(q * want, want);
            for (std::size_t i = 0; i < want; ++i) {
                if (i < found.size()) {
                    row_ids[i] = found[i].id;
                    row_distances[i] = found[i].distance;
                } else {  // fewer than k vectors in the index (or, rarely, reachable)
                    row_ids[i] = -1;
                    row_distances[i] = std::numeric_limits<float>::infinity();
                }
            }
        };
        {
            const py::gil_scoped_release release;
            const std::shared_lock lock(mutex_);
            const vecdex::MatrixView<float> rows = queries.view();
            if (threads == 1) {
                // Each result is written out before the next search, so none pile up.
                for (std::size_t q = 0; q < rows.rows(); ++q) {
                    write_row(q, index_.search(rows.row(q), want, breadth));
                }
            } else {
                // search_batch returns every result before any can be written out: about
                // n * k * 8 bytes held at once, the price of searching in parallel.
                const std::vector<std::vector<vecdex::Neighbor>> found =
                    index_.search_batch(rows, want, breadth, threads);
                for (std::size_t q = 0; q < found.size(); ++q) {
                    write_row(q, found[q]);
                }
            }
        }
        return {std::move(ids), std::move(distances)};
    }

    std::string repr() const {
        return std::format("<vecdex.Index space='{}' dim={} M={} ef_construction={} ef={} len={}>",
                           space(), dim(), m(), ef_construction(), ef_, size());
    }

private:
    HnswIndex index_;
    std::size_t ef_ = kDefaultEf;
    mutable std::shared_mutex mutex_;  // shared: reads of the index; exclusive: add_items
};

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "The C++ core of vecdex. Import the vecdex package, not this module.";
    m.attr("__version__") = VECDEX_VERSION;

    const vecdex::HnswParams defaults;
    py::class_<PyIndex>(m, "Index", R"doc(
        Approximate k-nearest-neighbor search over float32 vectors (an HNSW graph).

        Args:
            space: 'l2' (squared Euclidean distance), 'ip' (1 - inner product) or 'cosine'
                (1 - cosine similarity; vectors are normalized when stored).
            dim: the dimension of every vector.
            M: links per node on each layer (2 * M on layer 0). More links give better recall
                at the cost of memory and build time.
            ef_construction: the candidate-list size while building: build quality versus
                build time.
            seed: seeds the random layer of each node. The same vectors added in the same
                order with the same seed build the same graph.

        Input arrays must already be float32 and C-contiguous: they are read in place, and
        nothing is converted silently. Several threads may use one index: searches run in
        parallel with each other, and add_items runs alone. Both release the GIL. add_items and
        knn_query can also spread one call over several threads (num_threads).
    )doc")
        .def(py::init<std::string_view, std::int64_t, std::int64_t, std::int64_t, std::uint64_t>(),
             py::arg("space"), py::arg("dim"), py::arg("M") = defaults.M,
             py::arg("ef_construction") = defaults.ef_construction, py::arg("seed") = defaults.seed)
        .def("add_items", &PyIndex::add_items, py::arg("data"), py::kw_only(),
             py::arg("num_threads") = 1, R"doc(
            Adds vectors and returns their ids: consecutive int64 values from len(self) on.

            Args:
                data: a float32, C-contiguous array of shape (n, dim), or (dim,) for one vector.
                    It is read in place, not copied.
                num_threads: threads that link the batch into the graph; 0 means one per
                    hardware thread. With 1, the default, the graph is the one adding the
                    vectors one at a time builds, the same on every run. More threads build an
                    equally good graph faster, but it depends on how the threads interleave, so
                    it can differ from run to run.

            Raises TypeError if data isn't float32, and ValueError for a wrong shape, a
            non-contiguous or unaligned array, NaN or infinite values, under 'cosine' a zero
            vector, or a negative num_threads. The whole batch is checked before any of it is
            stored, so a rejected call adds nothing.
        )doc")
        .def("knn_query", &PyIndex::knn_query, py::arg("data"), py::arg("k") = 1,
             py::arg("ef") = py::none(), py::kw_only(), py::arg("num_threads") = 1, R"doc(
            Finds the k approximate nearest neighbors of each query.

            Args:
                data: queries, under the same rules as add_items' data.
                k: neighbors per query.
                ef: the search breadth, the recall-versus-speed dial; None uses self.ef.
                    Values below k act as k.
                num_threads: threads that share the queries; 0 means one per hardware thread.
                    The results are the same for any value. With 1, the default, the queries
                    run one after another on the calling thread.

            Returns:
                (ids, distances): int64 and float32 arrays of shape (n, k), each row closest
                first. Distances follow the space ('l2' is squared). If fewer than k vectors
                can be returned (k > len(self)), a row ends with id -1 and distance inf.
        )doc")
        .def_property("ef", &PyIndex::ef, &PyIndex::set_ef,
                      "knn_query's search breadth when its ef argument is None (default 64).")
        .def("set_ef", &PyIndex::set_ef, py::arg("ef"), "Sets self.ef.")
        .def_property_readonly("dim", &PyIndex::dim)
        .def_property_readonly("space", &PyIndex::space)
        .def_property_readonly("M", &PyIndex::m)
        .def_property_readonly("ef_construction", &PyIndex::ef_construction)
        .def("memory_bytes", &PyIndex::memory_bytes,
             "Bytes held by the vectors and graph links, not counting allocator overhead.")
        .def("__len__", &PyIndex::size)
        .def("__repr__", &PyIndex::repr);
}
