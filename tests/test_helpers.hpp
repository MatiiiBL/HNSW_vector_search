#pragma once

#include <ostream>

#include "vecdex/neighbor.hpp"

namespace vecdex {

// Lets GoogleTest print a Neighbor in failure messages (found through argument-dependent
// lookup, so it must live in the same namespace as Neighbor).
inline void PrintTo(const Neighbor& n, std::ostream* os) {
    *os << "{id " << n.id << ", dist " << n.distance << "}";
}

}  // namespace vecdex
