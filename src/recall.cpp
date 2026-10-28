#include "vecdex/recall.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>
#include <utility>

namespace vecdex {

double recall_at_k(std::span<const Neighbor> found, std::span<const std::int32_t> truth,
                   std::size_t k) {
    if (k == 0) {
        throw std::invalid_argument("recall_at_k: k must be positive");
    }
    if (truth.size() < k) {
        throw std::invalid_argument(
            std::format("recall_at_k: truth has {} ids, fewer than k = {}", truth.size(), k));
    }

    const auto top_found = found.first(std::min(k, found.size()));
    std::size_t hits = 0;
    // O(k^2) membership checks: trivial for the k used in practice (10 or 100), no allocation,
    // and each true id is counted at most once even if `found` repeats an id.
    for (const std::int32_t true_id : truth.first(k)) {
        const bool present = std::ranges::any_of(top_found, [true_id](const Neighbor& n) {
            return std::cmp_equal(n.id, true_id);  // uint32 vs int32, compared by value
        });
        if (present) {
            ++hits;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(k);
}

double threshold_recall_at_k(std::span<const Neighbor> found, float kth_true_distance,
                             std::size_t k, float tolerance) {
    if (k == 0) {
        throw std::invalid_argument("threshold_recall_at_k: k must be positive");
    }
    if (!std::isfinite(kth_true_distance) || !std::isfinite(tolerance) || tolerance < 0.0F) {
        throw std::invalid_argument(
            "threshold_recall_at_k: distance and tolerance must be finite, tolerance >= 0");
    }
    const float threshold =
        kth_true_distance + tolerance * std::max(std::abs(kth_true_distance), 1.0F);
    const auto top_found = found.first(std::min(k, found.size()));
    const auto hits = std::ranges::count_if(
        top_found, [threshold](const Neighbor& n) { return n.distance <= threshold; });
    return static_cast<double>(hits) / static_cast<double>(k);
}

}  // namespace vecdex
