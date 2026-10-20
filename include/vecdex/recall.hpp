#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "vecdex/neighbor.hpp"

namespace vecdex {

// recall@k = |first k of found ∩ first k of truth| / k, in [0, 1].
// `truth` holds the true nearest-neighbor ids, best first (e.g. a row of an .ivecs ground-truth
// file) and must have at least k entries; its ids are assumed distinct. A duplicate id in
// `found` counts once. Throws std::invalid_argument if k == 0 or truth.size() < k.
double recall_at_k(std::span<const Neighbor> found, std::span<const std::int32_t> truth,
                   std::size_t k);

}  // namespace vecdex
