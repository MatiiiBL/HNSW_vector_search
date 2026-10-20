#include "vecdex/distance.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using vecdex::l2_squared;

// Exact float comparisons are safe in this file: every value and partial sum is a small
// integer, which float represents exactly (below 2^24).

TEST(L2Squared, KnownValues) {
    const std::vector<float> origin{0.0F, 0.0F};
    const std::vector<float> p{3.0F, 4.0F};
    EXPECT_EQ(l2_squared(origin, p), 25.0F);
    EXPECT_EQ(l2_squared(p, origin), 25.0F);
    EXPECT_EQ(l2_squared(p, p), 0.0F);
}

TEST(L2Squared, EmptyVectorsAreAtDistanceZero) {
    EXPECT_EQ(l2_squared({}, {}), 0.0F);
}

// SIFT-like input: 128 small integers stored as floats.
TEST(L2Squared, MatchesIntegerArithmeticAtDim128) {
    constexpr std::size_t kDim = 128;
    std::vector<float> a(kDim);
    std::vector<float> b(kDim);
    std::int64_t expected = 0;
    for (std::size_t i = 0; i < kDim; ++i) {
        const auto x = static_cast<std::int64_t>(i);
        const std::int64_t y = 127 - x;
        a[i] = static_cast<float>(x);
        b[i] = static_cast<float>(y);
        expected += (x - y) * (x - y);
    }
    EXPECT_EQ(l2_squared(a, b), static_cast<float>(expected));
}

}  // namespace
