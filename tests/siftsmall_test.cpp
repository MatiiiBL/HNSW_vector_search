// End-to-end checks on siftsmall (10K base vectors, 100 queries, dim 128). The dataset is not
// in the repo; without it these tests are skipped, not failed. See the README to download it.

#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>

#include "vecdex/brute_force.hpp"
#include "vecdex/io.hpp"
#include "vecdex/recall.hpp"

namespace {

namespace fs = std::filesystem;

fs::path dataset_dir() {
    return fs::path(VECDEX_DATA_DIR) / "siftsmall";
}

bool dataset_present() {
    const fs::path dir = dataset_dir();
    return fs::exists(dir / "siftsmall_base.fvecs") && fs::exists(dir / "siftsmall_query.fvecs") &&
           fs::exists(dir / "siftsmall_groundtruth.ivecs");
}

class SiftSmall : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dataset_present()) {
            GTEST_SKIP() << "siftsmall not found in " << dataset_dir() << " (see README)";
        }
    }
};

TEST_F(SiftSmall, FilesHaveTheDocumentedShape) {
    const auto base = vecdex::read_fvecs(dataset_dir() / "siftsmall_base.fvecs");
    const auto queries = vecdex::read_fvecs(dataset_dir() / "siftsmall_query.fvecs");
    const auto truth = vecdex::read_ivecs(dataset_dir() / "siftsmall_groundtruth.ivecs");
    EXPECT_EQ(base.rows(), 10'000U);
    EXPECT_EQ(base.dim(), 128U);
    EXPECT_EQ(queries.rows(), 100U);
    EXPECT_EQ(queries.dim(), 128U);
    EXPECT_EQ(truth.rows(), 100U);
    EXPECT_EQ(truth.dim(), 100U);  // the 100 true nearest neighbors of each query
}

// M1's acceptance test: exact search must reproduce the published ground truth, for every
// query, at k = 10 and at k = 100 (all the ground truth there is).
TEST_F(SiftSmall, BruteForceReproducesGroundTruth) {
    const auto base = vecdex::read_fvecs(dataset_dir() / "siftsmall_base.fvecs");
    const auto queries = vecdex::read_fvecs(dataset_dir() / "siftsmall_query.fvecs");
    const auto truth = vecdex::read_ivecs(dataset_dir() / "siftsmall_groundtruth.ivecs");

    vecdex::BruteForceIndex index(base.dim());
    index.add(base);

    for (const std::size_t k : {std::size_t{10}, std::size_t{100}}) {
        double total = 0.0;
        for (std::size_t q = 0; q < queries.rows(); ++q) {
            const auto found = index.search(queries.row(q), k);
            const double recall = vecdex::recall_at_k(found, truth.row(q), k);
            EXPECT_DOUBLE_EQ(recall, 1.0) << "k " << k << ", query " << q;
            total += recall;
        }
        EXPECT_DOUBLE_EQ(total / static_cast<double>(queries.rows()), 1.0) << "mean recall@" << k;
    }
}

}  // namespace
