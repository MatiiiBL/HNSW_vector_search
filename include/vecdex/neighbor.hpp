#pragma once

#include <cstdint>

namespace vecdex {

// Identifies a stored vector. Ids are assigned 0, 1, 2, ... in insertion order.
// 32 bits halves the memory of graph links compared with size_t (matters from M3 on)
// and caps an index at about 4.29 billion vectors.
using Id = std::uint32_t;

// One search result.
struct Neighbor {
    float distance;  // to the query, under the index's Metric: smaller is closer
    Id id;

    friend bool operator==(const Neighbor&, const Neighbor&) = default;
};

// The ranking used by every index: smaller distance first, ties broken by smaller id so
// results are deterministic. This is a strict weak ordering as long as no distance is NaN,
// which is why the indexes reject non-finite input.
constexpr bool closer(const Neighbor& a, const Neighbor& b) noexcept {
    return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
}

}  // namespace vecdex
