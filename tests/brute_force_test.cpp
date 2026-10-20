#include "vecdex/brute_force.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
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

}  // namespace
