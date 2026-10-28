#include "vecdex/matrix.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using vecdex::Matrix;
using vecdex::MatrixView;

TEST(Matrix, RejectsZeroDim) {
    EXPECT_THROW((void)Matrix<float>(0), std::invalid_argument);
    EXPECT_THROW((void)Matrix<float>(3, 0), std::invalid_argument);
}

TEST(Matrix, StartsEmptyWhenGivenOnlyDim) {
    const Matrix<float> m(5);
    EXPECT_EQ(m.rows(), 0U);
    EXPECT_EQ(m.dim(), 5U);
    EXPECT_TRUE(m.empty());
}

TEST(Matrix, ZeroInitializesAndReportsShape) {
    const Matrix<float> m(3, 4);
    EXPECT_EQ(m.rows(), 3U);
    EXPECT_EQ(m.dim(), 4U);
    ASSERT_EQ(m.data().size(), 12U);
    for (const float v : m.data()) {
        EXPECT_EQ(v, 0.0F);
    }
}

TEST(Matrix, RowsAreContiguousAndRowMajor) {
    Matrix<int> m(3, 2);
    for (std::size_t i = 0; i < m.data().size(); ++i) {
        m.data()[i] = static_cast<int>(i);
    }
    for (std::size_t r = 0; r < m.rows(); ++r) {
        EXPECT_EQ(m.row(r).data(), m.data().data() + r * m.dim());
        EXPECT_EQ(m.row(r).size(), 2U);
    }
    EXPECT_EQ(m.row(1)[0], 2);
    EXPECT_EQ(m.row(2)[1], 5);
}

TEST(Matrix, AppendAddsWholeRows) {
    Matrix<int> m(2);
    m.append(std::vector<int>{1, 2});
    m.append(std::vector<int>{3, 4, 5, 6});  // two rows at once
    ASSERT_EQ(m.rows(), 3U);
    EXPECT_EQ(m.row(0)[1], 2);
    EXPECT_EQ(m.row(2)[0], 5);
}

TEST(Matrix, AppendRejectsPartialRowsAndStaysUnchanged) {
    Matrix<int> m(2);
    m.append(std::vector<int>{1, 2});
    EXPECT_THROW(m.append(std::vector<int>{1, 2, 3}), std::invalid_argument);
    EXPECT_EQ(m.rows(), 1U);
}

// m.row(0) points into the buffer that each append may reallocate. A naive
// vector::insert would read freed memory here, which ASan reports.
TEST(Matrix, AppendFromItsOwnRowSurvivesReallocation) {
    Matrix<int> m(3);
    m.append(std::vector<int>{7, 8, 9});
    for (int i = 0; i < 100; ++i) {
        m.append(m.row(0));
    }
    ASSERT_EQ(m.rows(), 101U);
    for (std::size_t r = 0; r < m.rows(); ++r) {
        EXPECT_EQ(m.row(r)[0], 7);
        EXPECT_EQ(m.row(r)[2], 9);
    }
}

TEST(Matrix, RefusesSizesThatOverflow) {
    const std::size_t huge = std::numeric_limits<std::size_t>::max() / 2;
    EXPECT_THROW((void)Matrix<float>(huge, 4), std::length_error);
    Matrix<float> m(4);
    EXPECT_THROW(m.reserve_rows(huge), std::length_error);
}

TEST(Matrix, CopyIsDeep) {
    const Matrix<int> a(1, 2);
    Matrix<int> b = a;
    b.row(0)[0] = 42;
    EXPECT_EQ(a.row(0)[0], 0);
}

TEST(Matrix, MovedFromMatrixIsEmptyButKeepsDim) {
    Matrix<float> a(3, 2);
    const Matrix<float> b = std::move(a);
    EXPECT_EQ(b.rows(), 3U);
    // rows() is derived from the buffer, so the moved-from object is still consistent.
    EXPECT_EQ(a.rows(), 0U);  // NOLINT(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
    EXPECT_EQ(a.dim(), 2U);
}

TEST(MatrixView, ViewsAMatrixWithoutCopying) {
    Matrix<int> m(3, 2);
    m.row(1)[1] = 7;
    const MatrixView<int> view = m;  // implicit, like a span from a vector
    EXPECT_EQ(view.rows(), 3U);
    EXPECT_EQ(view.dim(), 2U);
    EXPECT_EQ(view.data().data(), m.data().data());
    EXPECT_EQ(view.row(1)[1], 7);
    EXPECT_EQ(view.row(2).data(), m.row(2).data());
}

TEST(MatrixView, SplitsABufferIntoRows) {
    const std::vector<float> buffer{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const MatrixView<float> view(buffer, 3);
    EXPECT_EQ(view.rows(), 2U);
    EXPECT_FALSE(view.empty());
    EXPECT_EQ(view.row(1)[0], 4.0F);
    EXPECT_TRUE(MatrixView<float>(std::span<const float>(), 3).empty());
}

TEST(MatrixView, RejectsZeroDimAndPartialRows) {
    const std::vector<float> buffer(10);
    EXPECT_THROW((void)MatrixView<float>(buffer, 0), std::invalid_argument);
    EXPECT_THROW((void)MatrixView<float>(buffer, 4), std::invalid_argument);  // 2.5 rows
}

}  // namespace
