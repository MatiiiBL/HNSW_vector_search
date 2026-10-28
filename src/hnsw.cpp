#include "vecdex/hnsw.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <exception>
#include <format>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include "search_scratch.hpp"
#include "vector_ops.hpp"

namespace vecdex {

namespace detail {

// The mutexes of a parallel add(). The lock order rules out deadlock: a thread takes `entry` (if
// at all) before any node lock, holds at most one node lock at a time, and takes no other lock
// of the index while holding it. So the owner of a node lock never waits for one, a thread waiting
// for `entry` holds none, and no cycle of threads waiting on each other can form.
struct BuildLocks {
    explicit BuildLocks(std::size_t nodes) : links(std::make_unique<std::mutex[]>(nodes)) {}

    // One per node, guarding its link lists on every layer. A fixed-size array: mutexes can't
    // be moved, so a std::vector of them couldn't grow anyway.
    std::unique_ptr<std::mutex[]> links;
    std::mutex entry;  // guards entry_point_ and max_level_
};

}  // namespace detail

namespace {

detail::IndexShape shape_of(const HnswIndex& index) {
    return {.metric = index.metric(), .dim = index.dim(), .size = index.size()};
}

HnswParams validated(const HnswParams& params) {
    if (params.M < 2) {
        throw std::invalid_argument("HnswIndex: M must be at least 2");
    }
    if (params.ef_construction == 0) {
        throw std::invalid_argument("HnswIndex: ef_construction must be positive");
    }
    return params;
}

// The comparator for a candidate heap whose front is the *closest* element. std heaps keep the
// greatest element in front, so the order is `closer` reversed.
constexpr bool farther(const Neighbor& a, const Neighbor& b) noexcept {
    return closer(b, a);
}

// Asks the CPU to start loading [p, p + bytes) into cache and returns at once. A hint: it never
// faults and never changes results. 64-byte steps match x86 cache lines; on Apple's 128-byte
// lines every other hint is redundant but cheap. GCC and Clang only, like the rest of the build.
void prefetch(const void* p, std::size_t bytes) noexcept {
    const auto* first = static_cast<const char*>(p);
    for (std::size_t offset = 0; offset < bytes; offset += 64) {
        __builtin_prefetch(first + offset);
    }
}

// num_threads as given, with 0 meaning one per hardware thread. hardware_concurrency() returns
// 0 when it can't tell.
std::size_t resolve_threads(std::size_t num_threads) noexcept {
    return num_threads != 0 ? num_threads : std::max(1U, std::thread::hardware_concurrency());
}

// Calls fn(i) for every i in [0, count) on up to `threads` threads, the calling thread among
// them. Each thread claims the next unclaimed i from a shared counter, so a thread on a slower
// (efficiency) core simply claims fewer; with a fixed split, everyone would wait for the slowest
// share. If fn throws, the remaining items are skipped and the first exception is rethrown here
// after every thread has stopped: escaping a std::thread, it would call std::terminate.
template <typename Fn>
void parallel_for(std::size_t count, std::size_t threads, const Fn& fn) {
    assert(threads >= 1);
    if (count == 0) {
        return;
    }
    threads = std::min(threads, count);
    // Relaxed is enough: the counter only hands out distinct items. Whatever fn reads was
    // written before the threads started, or is guarded by a lock of its own.
    std::atomic<std::size_t> next{0};
    std::exception_ptr error;
    std::mutex error_mutex;
    const auto work = [&] {
        try {
            for (std::size_t i = next.fetch_add(1, std::memory_order_relaxed); i < count;
                 i = next.fetch_add(1, std::memory_order_relaxed)) {
                fn(i);
            }
        } catch (...) {
            next.store(count, std::memory_order_relaxed);  // everyone stops at their next claim
            const std::scoped_lock lock(error_mutex);
            if (!error) {
                error = std::current_exception();
            }
        }
    };
    std::vector<std::thread> helpers;
    helpers.reserve(threads - 1);
    for (std::size_t t = 1; t < threads; ++t) {
        try {
            helpers.emplace_back(work);
        } catch (...) {
            // The system won't start another thread (std::system_error). The threads already
            // running, this one included, still claim every item: the work gets done, slower.
            break;
        }
    }
    work();
    for (std::thread& helper : helpers) {
        helper.join();
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

}  // namespace

HnswIndex::HnswIndex(std::size_t dim, Metric metric, HnswParams params)
    : metric_(metric),
      params_(validated(params)),
      kernel_(metric == Metric::L2 ? kernels::active().l2_squared
                                   : kernels::active().inner_product),
      one_minus_(metric != Metric::L2),
      level_multiplier_(1.0 / std::log(static_cast<double>(params_.M))),
      rng_(params_.seed),
      vectors_(dim),
      scratch_(std::make_unique<detail::ScratchPool>()) {}

// Defined here, where ScratchPool and BuildLocks are complete types: unique_ptr's deleter needs
// them.
HnswIndex::~HnswIndex() = default;
HnswIndex::HnswIndex(HnswIndex&&) noexcept = default;
HnswIndex& HnswIndex::operator=(HnswIndex&&) noexcept = default;

// ---------------------------------------------------------------------------------------------
// Graph storage

const Id* HnswIndex::link_block(Id node, std::size_t layer) const noexcept {
    if (layer == 0) {
        return layer0_.data() + static_cast<std::size_t>(node) * (1 + max_links(0));
    }
    return upper_[node].data() + (layer - 1) * (1 + max_links(layer));
}

Id* HnswIndex::link_block(Id node, std::size_t layer) noexcept {
    return const_cast<Id*>(std::as_const(*this).link_block(node, layer));
}

std::span<const Id> HnswIndex::links(Id node, std::size_t layer) const noexcept {
    const Id* block = link_block(node, layer);
    return {block + 1, block[0]};
}

void HnswIndex::set_links(Id node, std::size_t layer,
                          std::span<const Neighbor> neighbors) noexcept {
    assert(neighbors.size() <= max_links(layer));
    Id* block = link_block(node, layer);
    block[0] = static_cast<Id>(neighbors.size());
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        block[1 + i] = neighbors[i].id;
    }
}

// While a parallel add() runs, a node's links are read or written only under its lock.
// Otherwise this returns a lock that owns no mutex, which compiles to nothing.
template <HnswIndex::Locking L>
std::unique_lock<std::mutex> HnswIndex::lock_links(Id node) const {
    if constexpr (L == Locking::PerNode) {
        return std::unique_lock(build_locks_->links[node]);
    } else {
        return {};
    }
}

// `node`'s links on `layer`. In a parallel add() another thread may rewrite them at any moment,
// so they are copied into `copy` under the node's lock, and the caller reads the copy.
template <HnswIndex::Locking L>
std::span<const Id> HnswIndex::read_links(Id node, std::size_t layer, std::vector<Id>& copy) const {
    if constexpr (L == Locking::PerNode) {
        const std::unique_lock<std::mutex> lock = lock_links<L>(node);
        const std::span<const Id> ids = links(node, layer);
        copy.assign(ids.begin(), ids.end());
        return copy;
    } else {
        return links(node, layer);
    }
}

float HnswIndex::distance(const float* a, const float* b) const noexcept {
    const float raw = kernel_(a, b, dim());
    return one_minus_ ? 1.0F - raw : raw;
}

std::size_t HnswIndex::memory_bytes() const noexcept {
    std::size_t bytes = vectors_.data().size_bytes() + levels_.size() * sizeof(std::uint8_t) +
                        layer0_.size() * sizeof(Id);
    for (const std::vector<Id>& blocks : upper_) {
        bytes += sizeof(std::vector<Id>) + blocks.size() * sizeof(Id);  // header + ids
    }
    return bytes;
}

// ---------------------------------------------------------------------------------------------
// Insertion (Algorithm 1 of the paper)

// level = floor(-ln(U) * mL) with U uniform in (0, 1]: P(level >= l) = M^-l, so each layer
// holds about 1/M of the nodes of the layer below. U comes from the top 53 bits of the
// generator rather than std::uniform_real_distribution, whose output differs between standard
// libraries; this way a given seed builds the same graph on every platform.
std::size_t HnswIndex::random_level() {
    const double u = 1.0 - static_cast<double>(rng_() >> 11U) * 0x1.0p-53;
    const auto level = static_cast<std::size_t>(-std::log(u) * level_multiplier_);
    assert(level <= 64);  // -ln(2^-53) / ln(2) is about 53, the worst case at M = 2
    return level;
}

// Appends one validated vector and empty link blocks for it; its id is size() - 1. Storage
// only grows here, never during insert(), so pointers into the link blocks stay valid there,
// even while several threads insert.
void HnswIndex::store(std::span<const float> vector) {
    if (!scratch_) {  // only after this index was moved from; search() relies on it existing
        scratch_ = std::make_unique<detail::ScratchPool>();
    }
    vectors_.append(vector);
    if (metric_ == Metric::Cosine) {
        detail::normalize(vectors_.row(size() - 1));
    }
    const std::size_t level = random_level();
    levels_.push_back(static_cast<std::uint8_t>(level));
    layer0_.resize(layer0_.size() + 1 + max_links(0), Id{0});
    upper_.emplace_back(level * (1 + max_links(1)), Id{0});
}

Id HnswIndex::add(std::span<const float> vector) {
    detail::check_new_vectors("HnswIndex::add", shape_of(*this), vector.size(), vector);
    store(vector);
    const auto id = static_cast<Id>(size() - 1);
    insert<Locking::None>(id);
    return id;
}

void HnswIndex::add(const Matrix<float>& vectors, std::size_t num_threads) {
    add(MatrixView<float>(vectors), num_threads);
}

void HnswIndex::add(MatrixView<float> vectors, std::size_t num_threads) {
    detail::check_new_vectors("HnswIndex::add", shape_of(*this), vectors.dim(), vectors.data());
    const std::size_t first = size();
    const std::size_t total = first + vectors.rows();
    vectors_.reserve_rows(total);
    levels_.reserve(total);
    layer0_.reserve(total * (1 + max_links(0)));
    upper_.reserve(total);
    const std::size_t threads = std::min(resolve_threads(num_threads), vectors.rows());
    if (threads <= 1) {
        for (std::size_t r = 0; r < vectors.rows(); ++r) {
            store(vectors.row(r));
            insert<Locking::None>(static_cast<Id>(size() - 1));
        }
        return;
    }

    // Store every vector before linking any, so no container grows while threads hold pointers
    // into it. The levels are drawn in the same order as in a serial build.
    build_locks_ = std::make_unique<detail::BuildLocks>(total);
    for (std::size_t r = 0; r < vectors.rows(); ++r) {
        store(vectors.row(r));
    }
    std::size_t next = first;
    if (next == 0) {  // the first node founds the graph; every other insert starts from it
        insert<Locking::None>(0);
        next = 1;
    }
    parallel_for(total - next, threads, [this, next](std::size_t i) {
        insert<Locking::PerNode>(static_cast<Id>(next + i));
    });
    build_locks_.reset();  // searches never lock, and a std::mutex is 40 to 64 bytes per node
}

// Links the already-stored node into the graph. With Locking::PerNode, other threads insert
// other nodes at the same time.
template <HnswIndex::Locking L>
void HnswIndex::insert(Id node) {
    const std::size_t level = levels_[node];
    if (node == 0) {  // the first node is the whole graph
        entry_point_ = node;
        max_level_ = level;
        return;
    }
    // A node that will raise the top layer keeps the entry lock until it is linked, so two such
    // nodes take turns and the second one links to the first on the new layers, as in hnswlib.
    // Each such insert raises max_level_, so this happens only a few times per build.
    std::unique_lock<std::mutex> entry_lock;
    if constexpr (L == Locking::PerNode) {
        entry_lock = std::unique_lock(build_locks_->entry);
    }
    const Id entry_point = entry_point_;
    const std::size_t max_level = max_level_;
    if (entry_lock.owns_lock() && level <= max_level) {
        entry_lock.unlock();
    }
    const detail::ScratchPool::Lease scratch = scratch_->acquire();
    const float* query = vector_of(node);

    // Above the new node's level, only walk greedily toward it.
    Neighbor entry{.distance = distance(query, vector_of(entry_point)), .id = entry_point};
    for (std::size_t layer = max_level; layer > level; --layer) {
        entry = greedy_closest<L>(query, entry, layer, *scratch);
    }

    // On every layer the node lives on, top-down: collect ef_construction candidates, choose the
    // best M of them, and start the next layer down from the closest one found. Fewer than M
    // candidates would leave the node under-linked, so ef is at least M (as in hnswlib).
    const std::size_t ef = std::max(params_.ef_construction, params_.M);
    const std::size_t top = std::min(level, max_level);
    std::vector<std::vector<Neighbor>> chosen(top + 1);
    for (std::size_t i = 0; i <= top; ++i) {
        const std::size_t layer = top - i;
        search_layer<L>(query, entry, ef, layer, *scratch);
        entry = *std::ranges::min_element(scratch->results, closer);
        chosen[layer] = select_neighbors(scratch->results, params_.M);
    }

    // Link only now: first the node's own lists, then the links back to it. Other threads can
    // reach the node only through a link back, so before the first one exists none can find it.
    // So no thread links to it while its own lists are unset (set_links would overwrite that
    // link), and its own searches above could not return the node itself. Serially the order
    // changes nothing: each layer's links are independent of the others.
    {
        const std::unique_lock<std::mutex> lock = lock_links<L>(node);
        for (std::size_t layer = 0; layer <= top; ++layer) {
            set_links(node, layer, chosen[layer]);
        }
    }
    for (std::size_t layer = 0; layer <= top; ++layer) {
        link_back<L>(node, layer, chosen[layer]);
    }

    if (level > max_level) {  // the new node towers over the graph: it becomes the entry
        max_level_ = level;
        entry_point_ = node;
    }
}

// Links each of `neighbors` back to `node` on `layer`. A neighbor whose list is already full
// re-selects its links from the old ones plus `node`, which keeps every degree bounded by
// max_links(layer). The neighbor's lock is held for the whole read-modify-write.
template <HnswIndex::Locking L>
void HnswIndex::link_back(Id node, std::size_t layer, std::span<const Neighbor> neighbors) {
    for (const Neighbor& neighbor : neighbors) {
        const std::unique_lock<std::mutex> lock = lock_links<L>(neighbor.id);
        Id* block = link_block(neighbor.id, layer);
        const std::size_t count = block[0];
        if (count < max_links(layer)) {
            block[1 + count] = node;
            block[0] = static_cast<Id>(count + 1);
            continue;
        }
        std::vector<Neighbor> candidates;
        candidates.reserve(count + 1);
        const float* base = vector_of(neighbor.id);
        for (const Id id : links(neighbor.id, layer)) {
            candidates.push_back({.distance = distance(base, vector_of(id)), .id = id});
        }
        candidates.push_back({.distance = neighbor.distance, .id = node});  // symmetric metric
        set_links(neighbor.id, layer, select_neighbors(std::move(candidates), max_links(layer)));
    }
}

// The paper's neighbor-selection heuristic (Algorithm 4), as hnswlib implements it.
// `candidates` carry their distance to the base node. Walking them from closest to farthest,
// keep one only if it is closer to the base than to every neighbor already kept. Links then
// spread out in different directions instead of all pointing into the nearest cluster, which
// keeps clustered data navigable. With m or fewer candidates, all are kept.
std::vector<Neighbor> HnswIndex::select_neighbors(std::vector<Neighbor> candidates,
                                                  std::size_t m) const {
    std::ranges::sort(candidates, closer);
    if (candidates.size() <= m) {
        return candidates;
    }
    std::vector<Neighbor> kept;
    kept.reserve(m);
    for (const Neighbor& candidate : candidates) {
        if (kept.size() == m) {
            break;
        }
        const float* v = vector_of(candidate.id);
        const bool diverse = std::ranges::none_of(kept, [&](const Neighbor& other) {
            return distance(v, vector_of(other.id)) < candidate.distance;
        });
        if (diverse) {
            kept.push_back(candidate);
        }
    }
    return kept;
}

// ---------------------------------------------------------------------------------------------
// Search (Algorithms 2 and 5)

// Moves from `entry` to whichever neighbor on `layer` is closer to the query, until none is.
// Used on the sparse upper layers, where one good starting point is all we need.
template <HnswIndex::Locking L>
Neighbor HnswIndex::greedy_closest(const float* query, Neighbor entry, std::size_t layer,
                                   detail::SearchScratch& scratch) const {
    for (bool moved = true; moved;) {
        moved = false;
        const Id current = entry.id;
        for (const Id id : read_links<L>(current, layer, scratch.link_copy)) {
            const float d = distance(query, vector_of(id));
            if (d < entry.distance) {
                entry = {.distance = d, .id = id};
                moved = true;
            }
        }
    }
    return entry;
}

// Best-first search on one layer. `candidates` is a min-heap of nodes still to expand (closest
// in front); `results` is a max-heap of the best ef found so far (worst in front). Stops when
// the closest unexpanded candidate is farther than the worst result while results are full:
// expanding it could only find nodes reachable through something already too far away.
// Leaves `scratch.results` as a max-heap under `closer`.
template <HnswIndex::Locking L>
void HnswIndex::search_layer(const float* query, Neighbor entry, std::size_t ef, std::size_t layer,
                             detail::SearchScratch& scratch) const {
    detail::VisitedList& visited = scratch.visited;
    std::vector<Neighbor>& candidates = scratch.candidates;
    std::vector<Neighbor>& results = scratch.results;
    std::vector<Id>& fresh = scratch.fresh;
    visited.reset(size());
    visited.insert(entry.id);
    candidates.assign(1, entry);  // assign keeps the capacity from earlier searches
    results.assign(1, entry);
    const std::size_t vector_bytes = dim() * sizeof(float);

    while (!candidates.empty()) {
        std::ranges::pop_heap(candidates, farther);
        const Neighbor current = candidates.back();
        candidates.pop_back();
        if (results.size() >= ef && closer(results.front(), current)) {
            break;
        }
        // Prefetch every unvisited neighbor's vector before computing any distance. Each one is
        // likely a cache miss (a random 512 bytes out of SIFT1M's 512 MB); issued together, the
        // misses overlap instead of each distance waiting for its own.
        fresh.clear();
        for (const Id id : read_links<L>(current.id, layer, scratch.link_copy)) {
            if (visited.insert(id)) {
                fresh.push_back(id);
                prefetch(vector_of(id), vector_bytes);
            }
        }
        for (const Id id : fresh) {
            const Neighbor next{.distance = distance(query, vector_of(id)), .id = id};
            if (results.size() < ef || closer(next, results.front())) {
                candidates.push_back(next);
                std::ranges::push_heap(candidates, farther);
                results.push_back(next);
                std::ranges::push_heap(results, closer);
                if (results.size() > ef) {
                    std::ranges::pop_heap(results, closer);
                    results.pop_back();
                }
            }
        }
    }
}

std::vector<Neighbor> HnswIndex::search(std::span<const float> query, std::size_t k,
                                        std::size_t ef) const {
    detail::check_query("HnswIndex::search", shape_of(*this), query, k);
    return search_unchecked(query, k, ef);
}

std::vector<std::vector<Neighbor>> HnswIndex::search_batch(MatrixView<float> queries, std::size_t k,
                                                           std::size_t ef,
                                                           std::size_t num_threads) const {
    std::vector<std::vector<Neighbor>> results(queries.rows());
    // Each query borrows its own scratch from the pool and writes only its own element of
    // `results`, so the threads share nothing but the graph, which nobody writes meanwhile.
    parallel_for(queries.rows(), resolve_threads(num_threads), [&](std::size_t q) {
        detail::check_query("HnswIndex::search_batch", shape_of(*this), queries.row(q), k);
        results[q] = search_unchecked(queries.row(q), k, ef);
    });
    return results;
}

// search() after the argument checks, which search_batch() runs on its worker threads.
std::vector<Neighbor> HnswIndex::search_unchecked(std::span<const float> query, std::size_t k,
                                                  std::size_t ef) const {
    if (size() == 0) {
        return {};
    }
    std::vector<float> normalized;  // used under Cosine only
    const float* q = detail::prepare_query(metric_, query, normalized).data();

    const detail::ScratchPool::Lease scratch = scratch_->acquire();
    Neighbor entry{.distance = distance(q, vector_of(entry_point_)), .id = entry_point_};
    for (std::size_t layer = max_level_; layer > 0; --layer) {
        entry = greedy_closest<Locking::None>(q, entry, layer, *scratch);
    }
    search_layer<Locking::None>(q, entry, std::max(ef, k), 0, *scratch);
    std::vector<Neighbor>& found = scratch->results;
    std::ranges::sort_heap(found, closer);  // closest first
    const auto count = static_cast<std::ptrdiff_t>(std::min(k, found.size()));
    return {found.begin(), found.begin() + count};  // a warm L2 search allocates only this
}

// ---------------------------------------------------------------------------------------------
// Diagnostics

GraphReport HnswIndex::inspect() const {
    GraphReport report;
    report.max_level = max_level_;
    const std::size_t n = size();
    if (n == 0) {
        return report;
    }
    const auto violation = [&report](std::string message) {
        report.violations.push_back(std::move(message));
    };

    std::size_t highest = 0;
    std::size_t layer0_links = 0;
    std::vector<Id> sorted;
    for (Id node = 0; node < n; ++node) {
        const std::size_t level = levels_[node];
        highest = std::max(highest, level);
        for (std::size_t layer = 0; layer <= level; ++layer) {
            const std::span<const Id> ids = links(node, layer);
            if (layer == 0) {
                layer0_links += ids.size();
            }
            if (ids.size() > max_links(layer)) {
                violation(std::format("node {} has {} links on layer {} (max {})", node, ids.size(),
                                      layer, max_links(layer)));
            }
            sorted.assign(ids.begin(), ids.end());
            std::ranges::sort(sorted);
            if (std::ranges::adjacent_find(sorted) != sorted.end()) {
                violation(std::format("node {} has a duplicate link on layer {}", node, layer));
            }
            for (const Id id : ids) {
                if (id == node) {
                    violation(std::format("node {} links to itself on layer {}", node, layer));
                } else if (id >= n) {
                    violation(std::format("node {} links to missing node {}", node, id));
                } else if (levels_[id] < layer) {
                    violation(std::format("node {} links to node {} on layer {}, above its level",
                                          node, id, layer));
                }
            }
        }
    }
    if (highest != max_level_) {
        violation(
            std::format("max level is {} but the tallest node reaches {}", max_level_, highest));
    }
    if (levels_[entry_point_] != max_level_) {
        violation(std::format("entry point {} is not on the top layer", entry_point_));
    }

    // Breadth-first search over layer-0 links from the entry point.
    std::vector<bool> seen(n, false);
    std::vector<Id> frontier{entry_point_};
    seen[entry_point_] = true;
    for (std::size_t i = 0; i < frontier.size(); ++i) {
        const Id current = frontier[i];
        for (const Id id : links(current, 0)) {
            if (id < n && !seen[id]) {
                seen[id] = true;
                frontier.push_back(id);
            }
        }
    }
    report.reachable = frontier.size();
    report.mean_degree0 = static_cast<double>(layer0_links) / static_cast<double>(n);
    return report;
}

}  // namespace vecdex
