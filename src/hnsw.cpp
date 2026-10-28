#include "vecdex/hnsw.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <stdexcept>
#include <utility>

#include "search_scratch.hpp"
#include "vector_ops.hpp"

namespace vecdex {
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

// Defined here, where ScratchPool is a complete type: unique_ptr's deleter needs it.
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
// only grows here, never during insert(), so pointers into the link blocks stay valid there.
void HnswIndex::store(std::span<const float> vector) {
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
    insert(id);
    return id;
}

void HnswIndex::add(const Matrix<float>& vectors) {
    detail::check_new_vectors("HnswIndex::add", shape_of(*this), vectors.dim(), vectors.data());
    const std::size_t total = size() + vectors.rows();
    vectors_.reserve_rows(total);
    levels_.reserve(total);
    layer0_.reserve(total * (1 + max_links(0)));
    upper_.reserve(total);
    for (std::size_t r = 0; r < vectors.rows(); ++r) {
        store(vectors.row(r));
        insert(static_cast<Id>(size() - 1));
    }
}

// Links the already-stored node into the graph.
void HnswIndex::insert(Id node) {
    if (!scratch_) {  // only after this index was moved from; search() relies on it existing
        scratch_ = std::make_unique<detail::ScratchPool>();
    }
    const std::size_t level = levels_[node];
    if (node == 0) {  // the first node is the whole graph
        entry_point_ = node;
        max_level_ = level;
        return;
    }
    const detail::ScratchPool::Lease scratch = scratch_->acquire();
    const float* query = vector_of(node);

    // Above the new node's level, only walk greedily toward it.
    Neighbor entry{.distance = distance(query, vector_of(entry_point_)), .id = entry_point_};
    for (std::size_t layer = max_level_; layer > level; --layer) {
        entry = greedy_closest(query, entry, layer);
    }

    // On every layer the node lives on, top-down: collect ef_construction candidates, link to
    // the best M of them, and start the next layer down from the closest one found. Fewer than
    // M candidates would leave the node under-linked, so ef is at least M (as in hnswlib).
    const std::size_t ef = std::max(params_.ef_construction, params_.M);
    const std::size_t top = std::min(level, max_level_);
    for (std::size_t i = 0; i <= top; ++i) {
        const std::size_t layer = top - i;
        search_layer(query, entry, ef, layer, *scratch);
        entry = *std::ranges::min_element(scratch->results, closer);
        connect(node, layer, select_neighbors(scratch->results, params_.M));
    }

    if (level > max_level_) {  // the new node towers over the graph: it becomes the entry
        max_level_ = level;
        entry_point_ = node;
    }
}

// Links `node` to `neighbors` on `layer`, and each neighbor back to `node`. A neighbor whose
// list is already full re-selects its links from the old ones plus `node`, which keeps every
// degree bounded by max_links(layer).
void HnswIndex::connect(Id node, std::size_t layer, std::span<const Neighbor> neighbors) {
    set_links(node, layer, neighbors);
    for (const Neighbor& neighbor : neighbors) {
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
Neighbor HnswIndex::greedy_closest(const float* query, Neighbor entry, std::size_t layer) const {
    for (bool moved = true; moved;) {
        moved = false;
        const Id current = entry.id;
        for (const Id id : links(current, layer)) {
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
void HnswIndex::search_layer(const float* query, Neighbor entry, std::size_t ef, std::size_t layer,
                             detail::SearchScratch& scratch) const {
    detail::VisitedList& visited = scratch.visited;
    std::vector<Neighbor>& candidates = scratch.candidates;
    std::vector<Neighbor>& results = scratch.results;
    visited.reset(size());
    visited.insert(entry.id);
    candidates.assign(1, entry);  // assign keeps the capacity from earlier searches
    results.assign(1, entry);

    while (!candidates.empty()) {
        std::ranges::pop_heap(candidates, farther);
        const Neighbor current = candidates.back();
        candidates.pop_back();
        if (results.size() >= ef && closer(results.front(), current)) {
            break;
        }
        for (const Id id : links(current.id, layer)) {
            if (!visited.insert(id)) {
                continue;
            }
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
    if (size() == 0) {
        return {};
    }
    std::vector<float> normalized;  // used under Cosine only
    const float* q = detail::prepare_query(metric_, query, normalized).data();

    Neighbor entry{.distance = distance(q, vector_of(entry_point_)), .id = entry_point_};
    for (std::size_t layer = max_level_; layer > 0; --layer) {
        entry = greedy_closest(q, entry, layer);
    }
    const detail::ScratchPool::Lease scratch = scratch_->acquire();
    search_layer(q, entry, std::max(ef, k), 0, *scratch);
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
