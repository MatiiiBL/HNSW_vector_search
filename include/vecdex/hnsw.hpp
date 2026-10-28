#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "vecdex/distance.hpp"
#include "vecdex/kernels.hpp"
#include "vecdex/matrix.hpp"
#include "vecdex/neighbor.hpp"

namespace vecdex {

namespace detail {
struct SearchScratch;
class ScratchPool;
struct BuildLocks;
}  // namespace detail

// Build parameters (Malkov & Yashunin, "Efficient and robust approximate nearest neighbor
// search using Hierarchical Navigable Small World graphs", TPAMI 2018).
struct HnswParams {
    // Links each new node makes per layer. Layer 0 allows up to 2*M links per node, higher
    // layers up to M. More links: better recall, more memory, slower build. Typical: 8 to 48.
    std::size_t M = 16;
    // Candidate-list size while inserting: build quality versus build time. Typical: 100 to 500.
    std::size_t ef_construction = 200;
    // Seeds the random level of each node: the same vectors added in the same order with the
    // same seed build the same graph, on every platform.
    std::uint64_t seed = 42;
};

// The structural health of the graph, from HnswIndex::inspect(). For tests and debugging.
struct GraphReport {
    std::vector<std::string> violations;  // broken invariants; empty for a healthy graph
    std::size_t reachable = 0;            // nodes reachable from the entry point on layer 0
    std::size_t max_level = 0;
    double mean_degree0 = 0.0;  // average number of layer-0 links per node
};

// Approximate k-nearest-neighbor search over a hierarchical navigable small-world graph.
//
// Every node lives on layer 0; a random, exponentially shrinking fraction also lives on the
// layers above, like the express lanes of a skip list. A search descends greedily from the
// top layer to find a good starting point, then explores layer 0 with a bounded best-first
// search whose breadth, ef, trades recall for speed.
//
// Thread safety: search() and search_batch() may run concurrently from several threads, as long
// as no add() runs at the same time. add() can link a batch of vectors on several threads itself.
// Errors: like BruteForceIndex, std::invalid_argument for bad arguments, and a rejected call
// leaves the index unchanged. If memory runs out partway through add(), the index may be left
// half-updated and should be discarded.
class HnswIndex {
public:
    // Throws std::invalid_argument if dim is 0, M < 2, or ef_construction is 0.
    explicit HnswIndex(std::size_t dim, Metric metric = Metric::L2, HnswParams params = {});
    ~HnswIndex();
    HnswIndex(HnswIndex&&) noexcept;
    HnswIndex& operator=(HnswIndex&&) noexcept;
    HnswIndex(const HnswIndex&) = delete;
    HnswIndex& operator=(const HnswIndex&) = delete;

    std::size_t dim() const noexcept { return vectors_.dim(); }
    std::size_t size() const noexcept { return vectors_.rows(); }
    Metric metric() const noexcept { return metric_; }
    const HnswParams& params() const noexcept { return params_; }

    // Stores and links one vector; returns its id.
    Id add(std::span<const float> vector);
    // Stores and links every row of `vectors`; they get consecutive ids starting at size(). All
    // rows are validated before any is stored. num_threads > 1 links them in parallel, and 0 means
    // one thread per hardware thread. A parallel build is as good but not reproducible, because
    // the graph depends on how the threads interleave; with one thread the graph is the same as
    // adding the rows one by one. The view lets callers add rows they own (a NumPy array, say)
    // without copying them into a Matrix first.
    void add(MatrixView<float> vectors, std::size_t num_threads = 1);
    void add(const Matrix<float>& vectors, std::size_t num_threads = 1);

    // Up to k approximate nearest neighbors of `query`, closest first, ties by smaller id.
    // `ef` is the size of the layer-0 candidate list, the recall-versus-speed dial. Values
    // below k are raised to k.
    std::vector<Neighbor> search(std::span<const float> query, std::size_t k, std::size_t ef) const;

    // search() for every row of `queries`, spread over num_threads threads (0: one per hardware
    // thread). Element i is exactly what search() returns for row i, whatever the thread count.
    std::vector<std::vector<Neighbor>> search_batch(MatrixView<float> queries, std::size_t k,
                                                    std::size_t ef,
                                                    std::size_t num_threads = 0) const;

    // Checks the graph's invariants and measures its connectivity. O(n * M), for tests.
    GraphReport inspect() const;

    // Bytes held by vectors and links (element storage only, no allocator overhead).
    std::size_t memory_bytes() const noexcept;

    // Writes the index to `path` in the format of docs/FORMAT.md. Atomic: the bytes go to a new
    // temporary file in the same directory, which replaces `path` only once it is complete, so
    // `path` never holds a partial index. Durable: the file and then the rename are flushed to
    // stable storage, so once save() returns, the new index survives a power loss. On failure
    // the temporary file is removed and `path` is untouched, unless only the last step, flushing
    // the directory, fails: then `path` holds the new index, which may not survive a power loss.
    // Like search(), it may run alongside searches but not alongside add().
    // Throws std::runtime_error, naming the path, on any I/O error.
    void save(const std::filesystem::path& path) const;

    // Reads an index written by save(), with everything search() and add() need. The file is
    // treated as hostile: its sizes are checked against the file size before anything is
    // allocated, a checksum catches corruption, and the graph is validated before any link is
    // followed. A bad file throws std::runtime_error naming the path; it never crashes.
    // add() after load() builds the same graph as add() on the index that was saved: the level
    // generator is restored by re-seeding it and skipping one draw per stored node.
    static HnswIndex load(const std::filesystem::path& path);

private:
    // Whether a graph walk locks each node's links. Only the inserts of a parallel add() do;
    // search() never overlaps add(), so it walks the graph without any synchronization.
    enum class Locking { None, PerNode };

    std::size_t max_links(std::size_t layer) const noexcept {
        return layer == 0 ? 2 * params_.M : params_.M;
    }
    const Id* link_block(Id node, std::size_t layer) const noexcept;
    Id* link_block(Id node, std::size_t layer) noexcept;
    std::span<const Id> links(Id node, std::size_t layer) const noexcept;
    void set_links(Id node, std::size_t layer, std::span<const Neighbor> neighbors) noexcept;
    template <Locking L>
    std::unique_lock<std::mutex> lock_links(Id node) const;
    template <Locking L>
    std::span<const Id> read_links(Id node, std::size_t layer, std::vector<Id>& copy) const;

    float distance(const float* a, const float* b) const noexcept;
    const float* vector_of(Id node) const noexcept { return vectors_.row(node).data(); }

    std::size_t random_level();
    void store(std::span<const float> vector);
    template <Locking L>
    void insert(Id node);
    template <Locking L>
    void link_back(Id node, std::size_t layer, std::span<const Neighbor> neighbors);
    std::vector<Neighbor> select_neighbors(std::vector<Neighbor> candidates, std::size_t m) const;
    template <Locking L>
    Neighbor greedy_closest(const float* query, Neighbor entry, std::size_t layer,
                            detail::SearchScratch& scratch) const;
    template <Locking L>
    void search_layer(const float* query, Neighbor entry, std::size_t ef, std::size_t layer,
                      detail::SearchScratch& scratch) const;
    std::vector<Neighbor> search_unchecked(std::span<const float> query, std::size_t k,
                                           std::size_t ef) const;

    // load()'s check that every link block can be followed safely (hnsw_io.cpp).
    void check_links(const std::filesystem::path& path) const;

    Metric metric_;
    HnswParams params_;
    kernels::DistanceFn kernel_;  // bound once: l2_squared, or inner_product for 1 - <a, b>
    bool one_minus_;
    double level_multiplier_;  // mL = 1 / ln(M): level = floor(-ln(U) * mL)
    std::mt19937_64 rng_;
    Matrix<float> vectors_;  // normalized under Metric::Cosine

    // The graph. Each link list is a fixed-size block [count, id_1, ..., id_max] in flat
    // storage, so following links doesn't chase per-node heap allocations.
    std::vector<std::uint8_t> levels_;    // highest layer of each node
    std::vector<Id> layer0_;              // one block of 1 + 2M ids per node
    std::vector<std::vector<Id>> upper_;  // per node: blocks of 1 + M ids for layers 1..level
    Id entry_point_ = 0;                  // a node on the top layer
    std::size_t max_level_ = 0;

    // Behind a pointer: the pool holds a mutex, which can't be moved, and the index must be.
    std::unique_ptr<detail::ScratchPool> scratch_;
    // The mutexes of a parallel add(), alive only while it runs.
    std::unique_ptr<detail::BuildLocks> build_locks_;
};

}  // namespace vecdex
