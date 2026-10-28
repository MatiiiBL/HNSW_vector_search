// Parallel construction (HnswIndex::add with num_threads > 1) and batch search. A parallel build
// depends on how its threads interleave, so these tests assert what every correct build has (no
// broken invariants, full reachability, recall no worse than a serial build's), never an exact
// graph. The tsan preset runs them under ThreadSanitizer to check for data races.

#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_helpers.hpp"
#include "vecdex/brute_force.hpp"
#include "vecdex/hnsw.hpp"
#include "vecdex/io.hpp"
#include "vecdex/recall.hpp"

namespace {

using vecdex::GraphReport;
using vecdex::HnswIndex;
using vecdex::HnswParams;
using vecdex::Matrix;
using vecdex::MatrixView;
using vecdex::Metric;
using vecdex::Neighbor;

// Vectors with integer components in [-128, 127], identical on every platform: mt19937's output
// is fully specified, unlike uniform_real_distribution's. Every squared L2 distance between them
// (at most 255^2 * dim, below 2^24 for dim <= 256) is exact in float whatever order a kernel sums
// in, so the same seed builds the same serial graph with every compiler and SIMD kernel.
Matrix<float> integer_matrix(std::size_t rows, std::size_t dim, std::uint32_t seed) {
    std::mt19937 rng(seed);
    Matrix<float> m(rows, dim);
    for (float& v : m.data()) {
        v = static_cast<float>(static_cast<int>(rng() >> 24U) - 128);  // the top 8 bits
    }
    return m;
}

// The ids of the k exact nearest neighbors of each query, laid out like a TEXMEX ground-truth
// file: one row per query, closest first.
Matrix<std::int32_t> exact_neighbors(const Matrix<float>& base, const Matrix<float>& queries,
                                     std::size_t k) {
    vecdex::BruteForceIndex exact(base.dim());
    exact.add(base);
    Matrix<std::int32_t> truth(queries.rows(), k);
    for (std::size_t q = 0; q < queries.rows(); ++q) {
        const std::vector<Neighbor> found = exact.search(queries.row(q), k);
        for (std::size_t i = 0; i < k; ++i) {
            truth.row(q)[i] = static_cast<std::int32_t>(found[i].id);
        }
    }
    return truth;
}

double mean_recall(const HnswIndex& index, const Matrix<float>& queries,
                   const Matrix<std::int32_t>& truth, std::size_t k, std::size_t ef) {
    const std::vector<std::vector<Neighbor>> found = index.search_batch(queries, k, ef);
    double total = 0.0;
    for (std::size_t q = 0; q < queries.rows(); ++q) {
        total += vecdex::recall_at_k(found[q], truth.row(q), k);
    }
    return total / static_cast<double>(queries.rows());
}

HnswIndex build(const Matrix<float>& base, const HnswParams& params, std::size_t threads,
                Metric metric = Metric::L2) {
    HnswIndex index(base.dim(), metric, params);
    index.add(base, threads);
    return index;
}

// The rows [first, first + count) of `m`.
Matrix<float> rows_of(const Matrix<float>& m, std::size_t first, std::size_t count) {
    Matrix<float> part(m.dim());
    part.append(m.data().subspan(first * m.dim(), count * m.dim()));
    return part;
}

// FNV-1a over the ids and distance bits of a list of search results.
std::uint64_t fingerprint(const std::vector<std::vector<Neighbor>>& results) {
    std::uint64_t hash = 14695981039346656037ULL;
    const auto mix = [&hash](std::uint32_t word) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            hash = (hash ^ ((word >> shift) & 0xFFU)) * 1099511628211ULL;
        }
    };
    for (const std::vector<Neighbor>& list : results) {
        for (const Neighbor& n : list) {
            mix(n.id);
            mix(std::bit_cast<std::uint32_t>(n.distance));
        }
    }
    return hash;
}

// No broken invariants, and at most 1 node in 1000 unreachable. HNSW doesn't guarantee
// reachability: pruning can orphan a node, serial builds included. In the calibration runs below,
// both kinds of build orphaned at most 2 nodes of 10,000.
void expect_healthy(const GraphReport& report, std::size_t nodes, const std::string& context) {
    EXPECT_TRUE(report.violations.empty()) << context << ": " << report.violations.front();
    EXPECT_GE(report.reachable, nodes - nodes / 1000) << context;
}

// Builds `base` on one thread and on 2, 4 and 8, and checks that each parallel build is healthy and
// its recall@10 at `ef` is at most 0.005 below the serial build's (the M6 acceptance bar).
void expect_parallel_matches_serial(const Matrix<float>& base, const Matrix<float>& queries,
                                    const HnswParams& params, std::size_t ef) {
    constexpr std::size_t kK = 10;
    const Matrix<std::int32_t> truth = exact_neighbors(base, queries, kK);
    const HnswIndex serial = build(base, params, 1);
    const double serial_recall = mean_recall(serial, queries, truth, kK, ef);
    ::testing::Test::RecordProperty("recall_serial", std::to_string(serial_recall));
    for (const std::size_t threads : {2U, 4U, 8U}) {
        const std::string context = std::to_string(threads) + " threads";
        const HnswIndex parallel = build(base, params, threads);
        const GraphReport report = parallel.inspect();
        expect_healthy(report, base.rows(), context);
        // Levels are drawn before the threads start, in the serial order.
        EXPECT_EQ(report.max_level, serial.inspect().max_level) << context;
        const double recall = mean_recall(parallel, queries, truth, kK, ef);
        ::testing::Test::RecordProperty("recall_threads" + std::to_string(threads),
                                        std::to_string(recall));
        EXPECT_GE(recall, serial_recall - 0.005) << context << ", serial " << serial_recall;
    }
}

// ---------------------------------------------------------------------------------------------
// Parallel add

// One thread must build exactly the M4 graph: it runs the same algorithm in the same order. The
// constants were recorded with the M4 code (commit 3051c93, before parallel add existed), with
// both the NEON and the scalar kernel. They hold on every platform because the data comes from
// integer_matrix (see there). k = ef = 1 makes the search a greedy walk, which changes with almost
// any change to the graph: every 4-thread build tried, and a build with seed 43, changed both.
TEST(HnswParallel, SerialBuildIsUnchanged) {
    const Matrix<float> base = integer_matrix(1000, 16, 1);
    const Matrix<float> queries = integer_matrix(200, 16, 2);
    const HnswParams params{.M = 8, .ef_construction = 64, .seed = 42};

    HnswIndex one_by_one(base.dim(), Metric::L2, params);
    for (std::size_t r = 0; r < base.rows(); ++r) {
        one_by_one.add(base.row(r));
    }
    const HnswIndex one_thread = build(base, params, 1);
    for (const HnswIndex* index : {&std::as_const(one_by_one), &one_thread}) {
        std::vector<std::vector<Neighbor>> greedy;
        std::vector<std::vector<Neighbor>> top10;
        for (std::size_t q = 0; q < queries.rows(); ++q) {
            greedy.push_back(index->search(queries.row(q), 1, 1));
            top10.push_back(index->search(queries.row(q), 10, 10));
        }
        const GraphReport report = index->inspect();
        EXPECT_EQ(fingerprint(greedy), 6133164406513967666ULL);
        EXPECT_EQ(fingerprint(top10), 15666754926265556722ULL);
        EXPECT_EQ(report.max_level, 3U);
        EXPECT_EQ(report.mean_degree0 * static_cast<double>(base.rows()), 11157.0);  // links
    }
}

// The 0.005 bar against measured noise: a parallel build differs from the serial one only by
// which concurrent inserts could see each other. Over 30 builds per thread count (Apple M4,
// 2026-10-27), the serial build scored 0.9606 at ef = 16 and the worst parallel one 0.9599, a gap
// of 0.0007. integer_matrix makes the data, and so the serial graph, the same on every platform.
TEST(HnswParallel, MatchesSerialRecallOnRandomData) {
    expect_parallel_matches_serial(integer_matrix(2000, 16, 1), integer_matrix(1000, 16, 2),
                                   {.M = 16, .ef_construction = 64, .seed = 42}, 16);
}

// Real, clustered data. The 100 published queries would make recall move in steps of 0.001 per
// missed neighbor, too coarse for a 0.005 bar, so this uses 500 vectors of the learn set as
// queries, with exact search as ground truth. Over 20 builds per thread count (Apple M4,
// 2026-10-27), the serial build scored 0.9852 at ef = 32 and the worst parallel one 0.9838, a gap
// of 0.0014. SIFT's components are integers, so the serial graph is the same on every platform.
TEST(HnswParallel, MatchesSerialRecallOnSiftSmall) {
    const std::filesystem::path dir = std::filesystem::path(VECDEX_DATA_DIR) / "siftsmall";
    if (!std::filesystem::exists(dir / "siftsmall_base.fvecs") ||
        !std::filesystem::exists(dir / "siftsmall_learn.fvecs")) {
        GTEST_SKIP() << "siftsmall not found in " << dir << " (see README)";
    }
    const Matrix<float> learn = vecdex::read_fvecs(dir / "siftsmall_learn.fvecs");
    expect_parallel_matches_serial(vecdex::read_fvecs(dir / "siftsmall_base.fvecs"),
                                   rows_of(learn, 0, 500),
                                   {.M = 16, .ef_construction = 64, .seed = 42}, 32);
}

// Parallel add appends to whatever the index holds, and a serial add can follow it.
TEST(HnswParallel, ExtendsAnExistingIndex) {
    constexpr std::size_t kK = 10;
    const Matrix<float> base = integer_matrix(2000, 16, 1);
    const Matrix<float> queries = integer_matrix(200, 16, 2);
    const Matrix<std::int32_t> truth = exact_neighbors(base, queries, kK);

    HnswIndex index(base.dim(), Metric::L2, {.M = 8, .ef_construction = 64});
    index.add(rows_of(base, 0, 500));
    index.add(rows_of(base, 500, 1500), 0);  // 0: one thread per hardware thread
    ASSERT_EQ(index.size(), base.rows());
    expect_healthy(index.inspect(), base.rows(), "after the parallel add");
    EXPECT_GE(mean_recall(index, queries, truth, kK, 64), 0.95);

    const std::vector<float> extra(base.dim(), 127.0F);  // a corner no random row sits on
    EXPECT_EQ(index.add(extra), base.rows());
    EXPECT_EQ(index.search(extra, 1, 10).front().id, base.rows());
    expect_healthy(index.inspect(), base.rows() + 1, "after a serial add");
}

TEST(HnswParallel, RejectedBatchLeavesTheIndexUnchanged) {
    HnswIndex index(4);
    index.add(integer_matrix(10, 4, 1), 4);
    Matrix<float> batch = integer_matrix(100, 4, 2);
    batch.row(50)[1] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(index.add(batch, 4), std::invalid_argument);
    EXPECT_EQ(index.size(), 10U);
    EXPECT_TRUE(index.inspect().violations.empty());
}

// add() is chosen by the argument's type, never by the argument count alone: a flat buffer of
// floats is one vector, and a batch must state its row width (a Matrix or a MatrixView). An
// ambiguous call would fail to compile here.
template <typename... Args>
using AddResult = decltype(std::declval<HnswIndex&>().add(std::declval<Args>()...));

template <typename... Args>
concept CanAdd =
    requires(HnswIndex& index, Args&&... args) { index.add(std::forward<Args>(args)...); };

static_assert(std::is_same_v<AddResult<std::span<const float>>, vecdex::Id>);
static_assert(std::is_same_v<AddResult<std::vector<float>>, vecdex::Id>);
static_assert(std::is_same_v<AddResult<const Matrix<float>&>, void>);
static_assert(std::is_same_v<AddResult<const Matrix<float>&, std::size_t>, void>);
static_assert(std::is_same_v<AddResult<MatrixView<float>>, void>);
static_assert(std::is_same_v<AddResult<MatrixView<float>, std::size_t>, void>);
static_assert(!CanAdd<std::vector<float>, std::size_t>);  // a flat buffer is never a batch
static_assert(!CanAdd<std::span<const float>, std::size_t>);

TEST(HnswParallel, BraceInitCallsPickTheIntendedAdd) {
    HnswIndex index(2);
    const std::vector<float> one{1.0F, 2.0F};
    EXPECT_EQ(index.add({one.data(), one.size()}), 0U);  // a span: one vector, returns its id
    const std::vector<float> two{3.0F, 4.0F, 5.0F, 6.0F};
    index.add({two, 2}, 2);  // a MatrixView: two rows of 2
    index.add(Matrix<float>(2, 2), 2);
    EXPECT_EQ(index.size(), 5U);
}

// Rows the caller owns, added without copying them into a Matrix (what the Python bindings need).
// As for every add(), all rows are validated before any is stored.
TEST(HnswParallel, AddsRowsFromAView) {
    const Matrix<float> base = integer_matrix(300, 8, 1);
    const std::vector<float> owned(base.data().begin(), base.data().end());
    HnswIndex index(8);
    index.add(MatrixView<float>(owned, 8), 4);
    ASSERT_EQ(index.size(), base.rows());
    expect_healthy(index.inspect(), base.rows(), "view");
    for (std::size_t r = 0; r < base.rows(); r += 37) {  // ef = n: exhaustive on a connected graph
        EXPECT_EQ(index.search(base.row(r), 1, base.rows()).front().id, r);
    }

    const std::vector<float> partial(20);  // 2.5 rows of 8: not even a valid view
    EXPECT_THROW(index.add(MatrixView<float>(partial, 8), 4), std::invalid_argument);
    EXPECT_THROW(index.add(MatrixView<float>(owned, 4), 4), std::invalid_argument);  // wrong width
    EXPECT_EQ(index.size(), base.rows());
}

// Identical vectors are the worst case for contention: all at distance 0 from one another, they
// link into the same few nodes, whose full lists every insert then re-selects. A small M makes
// lists overflow sooner. Runs under ThreadSanitizer like everything else.
TEST(HnswParallel, HandlesDuplicatesUnderContention) {
    const std::vector<float> same{0.5F, -0.5F, 0.25F, 1.0F};
    Matrix<float> base(same.size());
    for (int i = 0; i < 200; ++i) {
        base.append(same);
    }
    base.append(integer_matrix(200, same.size(), 6).data());
    const HnswIndex index = build(base, {.M = 4, .ef_construction = 16}, 8);
    const GraphReport report = index.inspect();
    EXPECT_TRUE(report.violations.empty()) << report.violations.front();
    const std::vector<Neighbor> found = index.search(same, 10, 50);
    ASSERT_EQ(found.size(), 10U);
    for (const Neighbor& n : found) {
        EXPECT_EQ(n.distance, 0.0F);
        EXPECT_LT(n.id, 200U);
    }
}

// ---------------------------------------------------------------------------------------------
// Batch search

TEST(HnswBatch, MatchesSequentialSearches) {
    const Matrix<float> base = integer_matrix(1000, 16, 1);
    const Matrix<float> queries = integer_matrix(100, 16, 3);
    for (const Metric metric : {Metric::L2, Metric::Cosine}) {
        const HnswIndex index = build(base, {.M = 8, .ef_construction = 64}, 1, metric);
        std::vector<std::vector<Neighbor>> expected;
        expected.reserve(queries.rows());
        for (std::size_t q = 0; q < queries.rows(); ++q) {
            expected.push_back(index.search(queries.row(q), 10, 32));
        }
        // 0 is one thread per hardware thread; 500 is more threads than queries.
        for (const std::size_t threads : {1U, 3U, 0U, 500U}) {
            EXPECT_EQ(index.search_batch(queries, 10, 32, threads), expected)
                << "metric " << static_cast<int>(metric) << ", " << threads << " threads";
        }
        const std::vector<float> owned(queries.data().begin(), queries.data().end());
        EXPECT_EQ(index.search_batch(MatrixView<float>(owned, queries.dim()), 10, 32, 2), expected);
    }
}

TEST(HnswBatch, HandlesEmptyBatchesAndEmptyIndexes) {
    const HnswIndex empty(4);
    const std::vector<std::vector<Neighbor>> results =
        empty.search_batch(integer_matrix(5, 4, 1), 3, 10, 2);
    ASSERT_EQ(results.size(), 5U);
    for (const std::vector<Neighbor>& found : results) {
        EXPECT_TRUE(found.empty());
    }
    const HnswIndex index = build(integer_matrix(50, 4, 2), {.M = 4, .ef_construction = 16}, 1);
    EXPECT_TRUE(index.search_batch(Matrix<float>(4), 3, 10).empty());
}

// An exception escaping a std::thread calls std::terminate. search_batch must instead stop its
// threads and rethrow on the caller.
TEST(HnswBatch, RethrowsWorkerErrorsOnTheCaller) {
    const HnswIndex index = build(integer_matrix(200, 4, 1), {.M = 8, .ef_construction = 32}, 1);
    Matrix<float> queries = integer_matrix(64, 4, 2);
    EXPECT_THROW(index.search_batch(queries, 0, 10, 4), std::invalid_argument);  // k = 0
    EXPECT_THROW(index.search_batch(integer_matrix(8, 3, 2), 5, 10, 4), std::invalid_argument);
    queries.row(37)[2] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(index.search_batch(queries, 5, 10, 4), std::invalid_argument);
}

}  // namespace
