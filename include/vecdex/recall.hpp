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

// Distance-threshold recall@k (the ann-benchmarks definition): the fraction of the first k
// results in `found` whose distance is at most the k-th true neighbor's distance, plus a
// tolerance. Unlike recall_at_k, a result that ties with a true neighbor but has another id
// counts as a hit, so an exact search scores 1.0 even when the ground truth broke ties
// differently. `kth_true_distance` must be in the same units as the found distances.
//
// The tolerance is relative, `tolerance * max(|kth_true_distance|, 1)`, and absorbs float
// rounding when two libraries sum the same distance in different orders: a float32 sum of d
// terms is off by at most about d * 2^-24 relative, 7.6e-6 at d = 128.
// Throws std::invalid_argument if k == 0, or if kth_true_distance or tolerance is not finite
// or tolerance is negative.
double threshold_recall_at_k(std::span<const Neighbor> found, float kth_true_distance,
                             std::size_t k, float tolerance = 1e-5F);

}  // namespace vecdex
