#include "vecdex/hnsw.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "test_helpers.hpp"
#include "vecdex/brute_force.hpp"
#include "vecdex/recall.hpp"

namespace {

using vecdex::HnswIndex;
using vecdex::HnswParams;
using vecdex::Matrix;
using vecdex::Metric;
using vecdex::Neighbor;

Matrix<float> random_matrix(std::size_t rows, std::size_t dim, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    Matrix<float> m(rows, dim);
    for (float& v : m.data()) {
        v = uniform(rng);
    }
    return m;
}

HnswParams params(std::size_t m, std::size_t ef_construction) {
    return {.M = m, .ef_construction = ef_construction, .seed = 42};
}

// Mean recall@k of `index` against exact search over the same vectors.
double mean_recall(const HnswIndex& index, const vecdex::BruteForceIndex& exact,
                   const Matrix<float>& queries, std::size_t k, std::size_t ef) {
    double total = 0.0;
    for (std::size_t q = 0; q < queries.rows(); ++q) {
        std::vector<std::int32_t> truth;
        for (const Neighbor& n : exact.search(queries.row(q), k)) {
            truth.push_back(static_cast<std::int32_t>(n.id));
        }
        total += vecdex::recall_at_k(index.search(queries.row(q), k, ef), truth, k);
    }
    return total / static_cast<double>(queries.rows());
}

// Both indexes over the same random vectors, for recall measurements.
struct Built {
    HnswIndex hnsw;
    vecdex::BruteForceIndex exact;
};

Built build(std::size_t rows, std::size_t dim, Metric metric, const HnswParams& p) {
    const Matrix<float> base = random_matrix(rows, dim, 1);
    Built built{HnswIndex(dim, metric, p), vecdex::BruteForceIndex(dim, metric)};
    built.hnsw.add(base);
    built.exact.add(base);
    return built;
}

TEST(Hnsw, RejectsBadParameters) {
    EXPECT_THROW((void)HnswIndex(0), std::invalid_argument);
    EXPECT_THROW((void)HnswIndex(4, Metric::L2, params(1, 100)), std::invalid_argument);
    EXPECT_THROW((void)HnswIndex(4, Metric::L2, params(16, 0)), std::invalid_argument);
}

TEST(Hnsw, EmptyIndexReturnsNothing) {
    const HnswIndex index(3);
    EXPECT_TRUE(index.search(std::vector<float>{1.0F, 2.0F, 3.0F}, 5, 10).empty());
    EXPECT_TRUE(index.inspect().violations.empty());
}

TEST(Hnsw, AssignsSequentialIdsAndFindsASingleVector) {
    HnswIndex index(3);
    EXPECT_EQ(index.add(std::vector<float>{1.0F, 2.0F, 3.0F}), 0U);
    const std::vector<Neighbor> expected{{0.0F, 0}};
    EXPECT_EQ(index.search(std::vector<float>{1.0F, 2.0F, 3.0F}, 1, 1), expected);
    EXPECT_EQ(index.add(std::vector<float>{0.0F, 0.0F, 0.0F}), 1U);
    EXPECT_EQ(index.size(), 2U);
}

TEST(Hnsw, ReturnsEverythingWhenKExceedsSize) {
    Built built = build(5, 4, Metric::L2, params(4, 10));
    const std::vector<float> query{0.1F, 0.2F, 0.3F, 0.4F};
    EXPECT_EQ(built.hnsw.search(query, 10, 10), built.exact.search(query, 10));
}

// With ef at least the number of nodes, the search can't stop early: it expands every
// reachable node, so on a fully connected graph the answer must be exactly brute force's.
TEST(Hnsw, ExactWhenEfCoversEveryNode) {
    constexpr std::size_t kRows = 300;
    Built built = build(kRows, 8, Metric::L2, params(8, 50));
    ASSERT_EQ(built.hnsw.inspect().reachable, kRows);
    const Matrix<float> queries = random_matrix(20, 8, 2);
    for (std::size_t q = 0; q < queries.rows(); ++q) {
        EXPECT_EQ(built.hnsw.search(queries.row(q), 10, kRows),
                  built.exact.search(queries.row(q), 10))
            << "query " << q;
    }
}

TEST(Hnsw, GraphInvariantsHold) {
    constexpr std::size_t kRows = 2000;
    constexpr std::size_t kM = 8;
    const Built built = build(kRows, 16, Metric::L2, params(kM, 64));
    const vecdex::GraphReport report = built.hnsw.inspect();
    EXPECT_TRUE(report.violations.empty()) << report.violations.front();
    EXPECT_EQ(report.reachable, kRows);
    EXPECT_GE(report.max_level, 1U);  // with 2000 nodes and M = 8, layer 1 is never empty
    EXPECT_LE(report.mean_degree0, 2.0 * kM);
    EXPECT_GE(report.mean_degree0, 1.0);
}

// Thresholds: measured on Apple clang/libc++ (2026-10-24): recall@10 = 0.883 / 0.999 / 1.000
// at ef = 10 / 40 / 160, and 1.000 for every metric at ef = 100. The random test vectors
// differ between standard libraries (uniform_real_distribution isn't bit-specified), so the
// bars sit well below those values. A broken comparator or neighbor heuristic falls far
// lower than these bars.
TEST(Hnsw, RecallIsHighAndGrowsWithEf) {
    const Built built = build(3000, 16, Metric::L2, params(16, 100));
    const Matrix<float> queries = random_matrix(100, 16, 3);
    const double low = mean_recall(built.hnsw, built.exact, queries, 10, 10);
    const double mid = mean_recall(built.hnsw, built.exact, queries, 10, 40);
    const double high = mean_recall(built.hnsw, built.exact, queries, 10, 160);
    RecordProperty("recall_ef10", std::to_string(low));
    RecordProperty("recall_ef40", std::to_string(mid));
    RecordProperty("recall_ef160", std::to_string(high));
    EXPECT_LE(low, mid);
    EXPECT_LE(mid, high);
    EXPECT_GE(mid, 0.95);
    EXPECT_GE(high, 0.99) << "low " << low << ", mid " << mid << ", high " << high;
}

TEST(Hnsw, WorksForEveryMetric) {
    for (const Metric metric : {Metric::L2, Metric::InnerProduct, Metric::Cosine}) {
        const Built built = build(1500, 16, metric, params(16, 100));
        const Matrix<float> queries = random_matrix(50, 16, 4);
        const double recall = mean_recall(built.hnsw, built.exact, queries, 10, 100);
        RecordProperty("recall_metric" + std::to_string(static_cast<int>(metric)),
                       std::to_string(recall));
        EXPECT_GE(recall, 0.95) << "metric " << static_cast<int>(metric) << ": " << recall;
        EXPECT_TRUE(built.hnsw.inspect().violations.empty());
    }
}

// ef_construction below M would leave each new node fewer than M candidates to link to (just
// one, here); like hnswlib, the index raises it to M. Without that, the mean degree is about 2
// and recall collapses. Even at ef_construction = M the build is so cheap that pruning
// orphaned 1 node of 1000 when measured (2026-10-24): HNSW doesn't guarantee reachability, so
// this asserts 99%.
TEST(Hnsw, TinyEfConstructionIsRaisedToM) {
    const Built built = build(1000, 8, Metric::L2, params(8, 1));
    const vecdex::GraphReport report = built.hnsw.inspect();
    EXPECT_GE(report.reachable, 990U);
    EXPECT_GE(report.mean_degree0, 8.0);
    const Matrix<float> queries = random_matrix(50, 8, 8);
    EXPECT_GE(mean_recall(built.hnsw, built.exact, queries, 10, 50), 0.95);
}

TEST(Hnsw, SameSeedBuildsTheSameGraph) {
    const Built a = build(1000, 8, Metric::L2, params(8, 50));
    const Built b = build(1000, 8, Metric::L2, params(8, 50));
    const Matrix<float> queries = random_matrix(20, 8, 5);
    for (std::size_t q = 0; q < queries.rows(); ++q) {
        EXPECT_EQ(a.hnsw.search(queries.row(q), 10, 20), b.hnsw.search(queries.row(q), 10, 20));
    }
    EXPECT_EQ(a.hnsw.inspect().max_level, b.hnsw.inspect().max_level);
}

// Duplicates are a classic stress case: every copy is at distance 0 from the others.
TEST(Hnsw, HandlesDuplicateVectors) {
    HnswIndex index(4, Metric::L2, params(8, 40));
    const std::vector<float> same{0.5F, -0.5F, 0.25F, 1.0F};
    for (int i = 0; i < 100; ++i) {
        index.add(same);
    }
    index.add(random_matrix(100, 4, 6));
    const vecdex::GraphReport report = index.inspect();
    EXPECT_TRUE(report.violations.empty()) << report.violations.front();
    const auto found = index.search(same, 10, 50);
    ASSERT_EQ(found.size(), 10U);
    for (const Neighbor& n : found) {
        EXPECT_EQ(n.distance, 0.0F);
        EXPECT_LT(n.id, 100U);
    }
}

TEST(Hnsw, RejectsBadArgumentsAndLeavesTheIndexUnchanged) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    HnswIndex index(2);
    index.add(std::vector<float>{1.0F, 2.0F});
    EXPECT_THROW(index.add(std::vector<float>{1.0F, 2.0F, 3.0F}), std::invalid_argument);
    EXPECT_THROW(index.add(std::vector<float>{nan, 0.0F}), std::invalid_argument);
    Matrix<float> batch(3, 2);
    batch.row(2)[0] = nan;
    EXPECT_THROW(index.add(batch), std::invalid_argument);
    EXPECT_EQ(index.size(), 1U);
    EXPECT_THROW(index.search(std::vector<float>{1.0F}, 1, 10), std::invalid_argument);
    EXPECT_THROW(index.search(std::vector<float>{1.0F, 2.0F}, 0, 10), std::invalid_argument);

    HnswIndex cosine(2, Metric::Cosine);
    EXPECT_THROW(cosine.add(std::vector<float>{0.0F, 0.0F}), std::invalid_argument);
    EXPECT_EQ(cosine.size(), 0U);
}

TEST(Hnsw, MovedFromIndexIsEmptyAndReusable) {
    HnswIndex a(2);
    a.add(std::vector<float>{1.0F, 0.0F});
    const HnswIndex b = std::move(a);
    EXPECT_EQ(b.size(), 1U);
    // NOLINTBEGIN(bugprone-use-after-move,clang-analyzer-cplusplus.Move): the point of the test
    EXPECT_EQ(a.size(), 0U);
    EXPECT_TRUE(a.search(std::vector<float>{1.0F, 1.0F}, 1, 10).empty());
    a.add(std::vector<float>{0.0F, 1.0F});  // one vector: the first insert takes a shortcut
    EXPECT_EQ(a.search(std::vector<float>{0.0F, 1.0F}, 1, 10).front().id, 0U);
    a.add(std::vector<float>{1.0F, 1.0F});
    EXPECT_EQ(a.search(std::vector<float>{1.0F, 1.0F}, 1, 10).front().id, 1U);
    // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
}

// Searches only read the graph and borrow scratch memory from a pool, so concurrent searches
// must give exactly the sequential answers. (M6 runs this under ThreadSanitizer.)
TEST(Hnsw, ConcurrentSearchesMatchSequentialOnes) {
    const Built built = build(1000, 8, Metric::L2, params(8, 50));
    const Matrix<float> queries = random_matrix(64, 8, 7);
    std::vector<std::vector<Neighbor>> expected;
    for (std::size_t q = 0; q < queries.rows(); ++q) {
        expected.push_back(built.hnsw.search(queries.row(q), 10, 40));
    }
    constexpr std::size_t kThreads = 4;
    std::vector<std::vector<std::vector<Neighbor>>> got(kThreads);
    // std::thread rather than std::jthread: Apple clang 17's libc++ (the macos-15 CI image) still
    // gates jthread behind -fexperimental-library.
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (std::size_t q = 0; q < queries.rows(); ++q) {
                got[t].push_back(built.hnsw.search(queries.row(q), 10, 40));
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    for (std::size_t t = 0; t < kThreads; ++t) {
        EXPECT_EQ(got[t], expected) << "thread " << t;
    }
}

TEST(Hnsw, MemoryAccountingCoversVectorsAndLinks) {
    constexpr std::size_t kRows = 500;
    constexpr std::size_t kDim = 8;
    constexpr std::size_t kM = 8;
    const Built built = build(kRows, kDim, Metric::L2, params(kM, 40));
    const std::size_t floor =
        kRows * kDim * sizeof(float) + kRows * (1 + 2 * kM) * sizeof(vecdex::Id);
    EXPECT_GE(built.hnsw.memory_bytes(), floor);
    EXPECT_LE(built.hnsw.memory_bytes(), 2 * floor);
}

}  // namespace
