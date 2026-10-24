#include "vecdex/brute_force.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "test_helpers.hpp"
#include "vecdex/distance.hpp"

namespace {

using vecdex::BruteForceIndex;
using vecdex::Matrix;
using vecdex::Neighbor;

// A 1-D index over `positions`, so expected distances can be checked by hand.
BruteForceIndex line_index(const std::vector<float>& positions) {
    BruteForceIndex index(1);
    for (const float p : positions) {
        index.add(std::vector<float>{p});
    }
    return index;
}

TEST(BruteForce, RejectsZeroDim) {
    EXPECT_THROW((void)BruteForceIndex(0), std::invalid_argument);
}

TEST(BruteForce, AddAssignsSequentialIds) {
    BruteForceIndex index(2);
    EXPECT_EQ(index.add(std::vector<float>{0.0F, 0.0F}), 0U);
    EXPECT_EQ(index.add(std::vector<float>{1.0F, 1.0F}), 1U);
    index.add(Matrix<float>(3, 2));
    EXPECT_EQ(index.size(), 5U);
}

TEST(BruteForce, FindsNearestOnALine) {
    // Squared distances from 2: id0 = 64, id1 = 4, id2 = 25, id3 = 1, id4 = 16.
    const auto index = line_index({10.0F, 0.0F, 7.0F, 3.0F, -2.0F});
    const std::vector<Neighbor> expected{{1.0F, 3}, {4.0F, 1}, {16.0F, 4}};
    EXPECT_EQ(index.search(std::vector<float>{2.0F}, 3), expected);
}

TEST(BruteForce, ReturnsEverythingWhenKExceedsSize) {
    const auto index = line_index({5.0F, 1.0F});
    const std::vector<Neighbor> expected{{1.0F, 1}, {25.0F, 0}};
    EXPECT_EQ(index.search(std::vector<float>{0.0F}, 10), expected);
}

TEST(BruteForce, EmptyIndexReturnsNothing) {
    const BruteForceIndex index(3);
    EXPECT_TRUE(index.search(std::vector<float>{1.0F, 2.0F, 3.0F}, 5).empty());
}

// Duplicates and mirror-image points all sit at distance 16 from the query: the result must
// not depend on scan or heap order.
TEST(BruteForce, BreaksTiesBySmallerId) {
    const auto index = line_index({4.0F, -4.0F, 4.0F, 9.0F, -4.0F});
    const std::vector<float> query{0.0F};
    const std::vector<Neighbor> top4{{16.0F, 0}, {16.0F, 1}, {16.0F, 2}, {16.0F, 4}};
    EXPECT_EQ(index.search(query, 4), top4);
    const std::vector<Neighbor> top2{{16.0F, 0}, {16.0F, 1}};
    EXPECT_EQ(index.search(query, 2), top2);
}

TEST(BruteForce, RejectsBadArguments) {
    BruteForceIndex index(2);
    index.add(std::vector<float>{1.0F, 2.0F});
    EXPECT_THROW(index.add(std::vector<float>{1.0F, 2.0F, 3.0F}), std::invalid_argument);
    EXPECT_THROW(index.add(Matrix<float>(1, 3)), std::invalid_argument);
    EXPECT_THROW(index.search(std::vector<float>{1.0F}, 1), std::invalid_argument);
    EXPECT_THROW(index.search(std::vector<float>{1.0F, 2.0F}, 0), std::invalid_argument);
    EXPECT_EQ(index.size(), 1U);
}

TEST(BruteForce, RejectsNonFiniteValues) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    BruteForceIndex index(2);
    EXPECT_THROW(index.add(std::vector<float>{nan, 0.0F}), std::invalid_argument);
    EXPECT_THROW(index.add(std::vector<float>{0.0F, -inf}), std::invalid_argument);
    index.add(std::vector<float>{0.0F, 0.0F});
    EXPECT_THROW(index.search(std::vector<float>{nan, 0.0F}, 1), std::invalid_argument);
}

TEST(BruteForce, RejectedBatchLeavesIndexUnchanged) {
    BruteForceIndex index(2);
    Matrix<float> batch(3, 2);
    batch.row(2)[1] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(index.add(batch), std::invalid_argument);
    EXPECT_EQ(index.size(), 0U);
}

// Property test on random data: for every k, the bounded heap must return exactly the first k
// entries of the full ranking.
//
// The reference ranking comes from the index itself (k = n), not from distances computed in
// this file. In Release builds the same l2_squared can round differently in two translation
// units (the optimizer fuses multiply-adds into FMA in one and not the other), so exact
// equality across files would fail for reasons unrelated to the search. The ranking itself
// is verified independently: complete, sorted, and with distances correct within a tolerance.
TEST(BruteForce, MatchesFullRankingOnRandomData) {
    constexpr std::size_t kDim = 16;
    constexpr std::size_t kBase = 500;
    constexpr std::size_t kQueries = 20;
    std::mt19937 rng(42);  // fixed seed: any failure reproduces exactly
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);

    Matrix<float> base(kBase, kDim);
    for (float& v : base.data()) {
        v = uniform(rng);
    }
    BruteForceIndex index(kDim);
    index.add(base);

    std::vector<vecdex::Id> every_id(kBase);
    std::iota(every_id.begin(), every_id.end(), vecdex::Id{0});

    std::vector<float> query(kDim);
    for (std::size_t q = 0; q < kQueries; ++q) {
        for (float& v : query) {
            v = uniform(rng);
        }
        const std::vector<Neighbor> ranking = index.search(query, kBase);

        ASSERT_TRUE(std::ranges::is_sorted(ranking, vecdex::closer)) << "query " << q;
        std::vector<vecdex::Id> ids;
        for (const Neighbor& n : ranking) {
            ids.push_back(n.id);
            const float expected = vecdex::l2_squared(query, base.row(n.id));
            EXPECT_NEAR(n.distance, expected, 1e-5F * expected) << "query " << q;
        }
        std::ranges::sort(ids);
        ASSERT_EQ(ids, every_id) << "query " << q << ": every id exactly once";

        for (const std::size_t k : std::array<std::size_t, 3>{1, 10, 100}) {
            const std::vector<Neighbor> top_k(ranking.begin(),
                                              ranking.begin() + static_cast<std::ptrdiff_t>(k));
            EXPECT_EQ(index.search(query, k), top_k) << "query " << q << ", k " << k;
        }
    }
}

TEST(BruteForce, ReportsItsMetric) {
    EXPECT_EQ(BruteForceIndex(3).metric(), vecdex::Metric::L2);
    EXPECT_EQ(BruteForceIndex(3, vecdex::Metric::Cosine).metric(), vecdex::Metric::Cosine);
}

TEST(BruteForceInnerProduct, RanksByLargestDotProduct) {
    BruteForceIndex index(2, vecdex::Metric::InnerProduct);
    index.add(std::vector<float>{1.0F, 0.0F});  // dot with the query: 1
    index.add(std::vector<float>{0.0F, 2.0F});  // 2
    index.add(std::vector<float>{3.0F, 3.0F});  // 6
    const std::vector<Neighbor> expected{{-5.0F, 2}, {-1.0F, 1}, {0.0F, 0}};  // 1 - dot
    EXPECT_EQ(index.search(std::vector<float>{1.0F, 1.0F}, 3), expected);
}

// All lengths here are exact in float (sqrt(100) = 10, ...), so exact comparison is safe.
TEST(BruteForceCosine, IgnoresVectorLength) {
    BruteForceIndex index(2, vecdex::Metric::Cosine);
    index.add(std::vector<float>{10.0F, 0.0F});  // same direction as the query, longer
    index.add(std::vector<float>{0.0F, 0.5F});   // orthogonal
    index.add(std::vector<float>{-3.0F, 0.0F});  // opposite
    const std::vector<Neighbor> expected{{0.0F, 0}, {1.0F, 1}, {2.0F, 2}};
    EXPECT_EQ(index.search(std::vector<float>{2.0F, 0.0F}, 3), expected);
}

TEST(BruteForceCosine, RejectsZeroVectors) {
    BruteForceIndex index(2, vecdex::Metric::Cosine);
    EXPECT_THROW(index.add(std::vector<float>{0.0F, 0.0F}), std::invalid_argument);
    Matrix<float> batch(2, 2);
    batch.row(0)[0] = 1.0F;  // row 1 stays all zeros
    EXPECT_THROW(index.add(batch), std::invalid_argument);
    EXPECT_EQ(index.size(), 0U);  // the valid row 0 wasn't stored either

    index.add(std::vector<float>{1.0F, 1.0F});
    EXPECT_THROW(index.search(std::vector<float>{0.0F, 0.0F}, 1), std::invalid_argument);
}

// The distance under `metric`, computed independently in double precision.
double reference_distance(vecdex::Metric metric, std::span<const float> a,
                          std::span<const float> b) {
    double l2 = 0.0;
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double x = a[i];
        const double y = b[i];
        l2 += (x - y) * (x - y);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    switch (metric) {
        case vecdex::Metric::L2:
            return l2;
        case vecdex::Metric::InnerProduct:
            return 1.0 - dot;
        case vecdex::Metric::Cosine:
            return 1.0 - dot / std::sqrt(norm_a * norm_b);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

// On random data, rounding can legitimately swap two near-tied neighbors, so this checks the
// defining property of exact top-k instead of one specific order: every returned vector is at
// least as close as the true k-th nearest (in double precision), up to a small tolerance.
TEST(BruteForce, IsExactUpToRoundingForEveryMetric) {
    constexpr std::size_t kDim = 24;
    constexpr std::size_t kBase = 300;
    constexpr std::size_t kQueries = 10;
    std::mt19937 rng(5);  // fixed seed: any failure reproduces exactly
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    Matrix<float> base(kBase, kDim);
    for (float& v : base.data()) {
        v = uniform(rng);
    }
    const auto tolerance = [](double reference) { return 1e-5 * (1.0 + std::abs(reference)); };

    for (const auto metric :
         {vecdex::Metric::L2, vecdex::Metric::InnerProduct, vecdex::Metric::Cosine}) {
        BruteForceIndex index(kDim, metric);
        index.add(base);
        std::vector<float> query(kDim);
        for (std::size_t q = 0; q < kQueries; ++q) {
            for (float& v : query) {
                v = uniform(rng);
            }
            std::vector<double> reference(kBase);
            for (std::size_t i = 0; i < kBase; ++i) {
                reference[i] = reference_distance(metric, query, base.row(i));
            }
            std::vector<double> sorted = reference;
            std::ranges::sort(sorted);

            for (const std::size_t k : std::array<std::size_t, 3>{1, 10, 50}) {
                const auto result = index.search(query, k);
                ASSERT_EQ(result.size(), k);
                const double kth = sorted[k - 1];
                std::vector<vecdex::Id> ids;
                for (const Neighbor& n : result) {
                    ids.push_back(n.id);
                    EXPECT_LE(reference[n.id], kth + tolerance(kth))
                        << "metric " << static_cast<int>(metric) << ", query " << q << ", k " << k;
                    EXPECT_NEAR(n.distance, reference[n.id], tolerance(reference[n.id]));
                }
                std::ranges::sort(ids);
                EXPECT_EQ(std::ranges::adjacent_find(ids), ids.end()) << "duplicate id";
            }
        }
    }
}

}  // namespace
