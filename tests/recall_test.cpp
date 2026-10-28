#include "vecdex/recall.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using vecdex::Neighbor;
using vecdex::recall_at_k;
using vecdex::threshold_recall_at_k;

// Results with the given ids, in order. Distances don't matter to recall.
std::vector<Neighbor> results(std::initializer_list<vecdex::Id> ids) {
    std::vector<Neighbor> out;
    float distance = 0.0F;
    for (const vecdex::Id id : ids) {
        out.push_back({.distance = distance, .id = id});
        distance += 1.0F;
    }
    return out;
}

TEST(RecallAtK, PerfectPartialAndZero) {
    const std::vector<std::int32_t> truth{4, 8, 15, 16};
    EXPECT_DOUBLE_EQ(recall_at_k(results({4, 8, 15, 16}), truth, 4), 1.0);
    EXPECT_DOUBLE_EQ(recall_at_k(results({16, 15, 8, 4}), truth, 4), 1.0);  // order is ignored
    EXPECT_DOUBLE_EQ(recall_at_k(results({4, 99, 15, 98}), truth, 4), 0.5);
    EXPECT_DOUBLE_EQ(recall_at_k(results({1, 2, 3, 5}), truth, 4), 0.0);
}

TEST(RecallAtK, UsesOnlyTheFirstKOfEachList) {
    const std::vector<std::int32_t> truth{4, 8, 15, 16};
    // Found top-2 is {4, 15}; true top-2 is {4, 8}. 15 is a true neighbor, but not a top-2 one.
    EXPECT_DOUBLE_EQ(recall_at_k(results({4, 15, 8}), truth, 2), 0.5);
}

TEST(RecallAtK, MissingResultsCountAsMisses) {
    const std::vector<std::int32_t> truth{1, 2, 3, 4};
    EXPECT_DOUBLE_EQ(recall_at_k(results({2, 1}), truth, 4), 0.5);
    EXPECT_DOUBLE_EQ(recall_at_k({}, truth, 4), 0.0);
}

TEST(RecallAtK, DuplicateResultsCountOnce) {
    const std::vector<std::int32_t> truth{7, 8, 9};
    EXPECT_DOUBLE_EQ(recall_at_k(results({7, 7, 7}), truth, 3), 1.0 / 3.0);
}

TEST(RecallAtK, RejectsBadArguments) {
    const std::vector<std::int32_t> truth{1, 2};
    EXPECT_THROW((void)recall_at_k(results({1}), truth, 0), std::invalid_argument);
    EXPECT_THROW((void)recall_at_k(results({1}), truth, 3), std::invalid_argument);
}

// Results with the given distances, in order. Ids don't matter to threshold recall.
std::vector<Neighbor> at_distances(std::initializer_list<float> distances) {
    std::vector<Neighbor> out;
    vecdex::Id id = 0;
    for (const float distance : distances) {
        out.push_back({.distance = distance, .id = id++});
    }
    return out;
}

TEST(ThresholdRecallAtK, CountsTiesWithTheKthNeighborAsHits) {
    // True top-3 distances are {1, 2, 5}. An exact search that returned another node at
    // distance 5 loses a point under id-based recall, but not here.
    EXPECT_DOUBLE_EQ(threshold_recall_at_k(at_distances({1, 2, 5}), 5.0F, 3), 1.0);
    EXPECT_DOUBLE_EQ(threshold_recall_at_k(at_distances({1, 2, 6}), 5.0F, 3), 2.0 / 3.0);
    EXPECT_DOUBLE_EQ(threshold_recall_at_k(at_distances({7, 8, 9}), 5.0F, 3), 0.0);
}

TEST(ThresholdRecallAtK, UsesOnlyTheFirstKAndCountsMissingResultsAsMisses) {
    EXPECT_DOUBLE_EQ(threshold_recall_at_k(at_distances({1, 9, 2}), 5.0F, 2), 0.5);
    EXPECT_DOUBLE_EQ(threshold_recall_at_k(at_distances({1}), 5.0F, 4), 0.25);
    EXPECT_DOUBLE_EQ(threshold_recall_at_k({}, 5.0F, 4), 0.0);
}

TEST(ThresholdRecallAtK, ToleranceIsRelativeAboveOneAndAbsoluteBelow) {
    // 1e5 * (1 + 1e-5) = 100001: a result 0.5 above the threshold distance is within it, 2.0
    // above is not. Both are exactly representable floats at this magnitude.
    EXPECT_DOUBLE_EQ(threshold_recall_at_k(at_distances({100000.5F}), 100000.0F, 1), 1.0);
    EXPECT_DOUBLE_EQ(threshold_recall_at_k(at_distances({100002.0F}), 100000.0F, 1), 0.0);
    // Below 1 the slack is an absolute 1e-5, so an exact duplicate (distance 0) still has some.
    EXPECT_DOUBLE_EQ(threshold_recall_at_k(at_distances({5e-6F}), 0.0F, 1), 1.0);
    EXPECT_DOUBLE_EQ(threshold_recall_at_k(at_distances({5e-5F}), 0.0F, 1), 0.0);
    EXPECT_DOUBLE_EQ(threshold_recall_at_k(at_distances({100000.5F}), 100000.0F, 1, 0.0F), 0.0);
}

TEST(ThresholdRecallAtK, RejectsBadArguments) {
    const auto found = at_distances({1});
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_THROW((void)threshold_recall_at_k(found, 1.0F, 0), std::invalid_argument);
    EXPECT_THROW((void)threshold_recall_at_k(found, nan, 1), std::invalid_argument);
    EXPECT_THROW((void)threshold_recall_at_k(found, inf, 1), std::invalid_argument);
    EXPECT_THROW((void)threshold_recall_at_k(found, 1.0F, 1, -1e-5F), std::invalid_argument);
    EXPECT_THROW((void)threshold_recall_at_k(found, 1.0F, 1, nan), std::invalid_argument);
}

}  // namespace
