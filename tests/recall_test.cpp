#include "vecdex/recall.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace {

using vecdex::Neighbor;
using vecdex::recall_at_k;

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

}  // namespace
