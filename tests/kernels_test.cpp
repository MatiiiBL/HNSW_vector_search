#include "vecdex/kernels.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using vecdex::kernels::DistanceFn;
using vecdex::kernels::Isa;
using vecdex::kernels::KernelSet;

// Every dim from 0 to 100 hits every tail length of the 4-, 8-, 16- and 32-wide loops. The
// rest are common embedding sizes and their neighbors.
std::vector<std::size_t> dims_to_test() {
    std::vector<std::size_t> dims;
    for (std::size_t d = 0; d <= 100; ++d) {
        dims.push_back(d);
    }
    constexpr std::array<std::size_t, 17> kEmbeddingSizes{
        127, 128, 129, 255, 256, 257, 383, 384, 385, 511, 512, 513, 767, 768, 769, 1023, 1024};
    dims.insert(dims.end(), kEmbeddingSizes.begin(), kEmbeddingSizes.end());
    return dims;
}

// A vector whose data starts one float into its allocation: misaligned for 16- and 32-byte
// SIMD loads, and ending exactly at the end of the allocation, so any read past the end is a
// heap-buffer-overflow under ASan.
class OffsetVector {
public:
    OffsetVector(std::size_t dim, std::mt19937& rng) : storage_(dim + 1) {
        std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
        for (std::size_t i = 1; i < storage_.size(); ++i) {
            storage_[i] = uniform(rng);
        }
    }
    const float* data() const { return storage_.data() + 1; }

private:
    std::vector<float> storage_;
};

// The exact result (in double precision) and the sum of |terms|. Any float summation order
// (sequential, SIMD lanes, with or without FMA) lands within about dim * epsilon * magnitude.
struct Reference {
    double value = 0.0;
    double magnitude = 0.0;
};

Reference l2_reference(const float* a, const float* b, std::size_t dim) {
    Reference r;
    for (std::size_t i = 0; i < dim; ++i) {
        const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        r.value += diff * diff;
    }
    r.magnitude = r.value;
    return r;
}

Reference inner_product_reference(const float* a, const float* b, std::size_t dim) {
    Reference r;
    for (std::size_t i = 0; i < dim; ++i) {
        const double term = static_cast<double>(a[i]) * static_cast<double>(b[i]);
        r.value += term;
        r.magnitude += std::abs(term);
    }
    return r;
}

double tolerance(std::size_t dim, double magnitude) {
    constexpr double kEpsilon = std::numeric_limits<float>::epsilon();
    return 2.0 * static_cast<double>(dim + 2) * kEpsilon * magnitude;
}

std::string label(const KernelSet& set, std::size_t dim) {
    return std::string(vecdex::kernels::name(set.isa)) + ", dim " + std::to_string(dim);
}

TEST(Kernels, ScalarIsAlwaysAvailableAndListedFirst) {
    const auto sets = vecdex::kernels::available();
    ASSERT_FALSE(sets.empty());
    EXPECT_EQ(sets.front().isa, Isa::Scalar);
}

TEST(Kernels, ActiveSetIsTheFastestUnlessOverridden) {
    if (std::getenv("VECDEX_ISA") != nullptr) {
        GTEST_SKIP() << "VECDEX_ISA overrides the choice";
    }
    EXPECT_EQ(vecdex::kernels::active().isa, vecdex::kernels::available().back().isa);
}

// CI sets VECDEX_EXPECT_ISA (avx2 on x86-64, neon on Apple Silicon), so a runner that silently
// lacks the SIMD path fails here instead of testing only the scalar kernels.
TEST(Kernels, ExpectedIsaIsAvailable) {
    const char* expected = std::getenv("VECDEX_EXPECT_ISA");
    if (expected == nullptr) {
        GTEST_SKIP() << "set VECDEX_EXPECT_ISA to require an instruction set";
    }
    bool found = false;
    for (const KernelSet& set : vecdex::kernels::available()) {
        found = found || vecdex::kernels::name(set.isa) == expected;
    }
    EXPECT_TRUE(found) << expected << " is not available on this machine";
}

TEST(Kernels, MatchDoublePrecisionReferenceAtEveryDim) {
    std::mt19937 rng(7);  // fixed seed: any failure reproduces exactly
    for (const std::size_t dim : dims_to_test()) {
        const OffsetVector a(dim, rng);
        const OffsetVector b(dim, rng);
        const Reference l2 = l2_reference(a.data(), b.data(), dim);
        const Reference ip = inner_product_reference(a.data(), b.data(), dim);
        for (const KernelSet& set : vecdex::kernels::available()) {
            EXPECT_NEAR(set.l2_squared(a.data(), b.data(), dim), l2.value,
                        tolerance(dim, l2.magnitude))
                << "l2_squared, " << label(set, dim);
            EXPECT_NEAR(set.inner_product(a.data(), b.data(), dim), ip.value,
                        tolerance(dim, ip.magnitude))
                << "inner_product, " << label(set, dim);
        }
    }
}

TEST(Kernels, IdenticalVectorsAreExactlyZeroApart) {
    std::mt19937 rng(11);
    for (const std::size_t dim : dims_to_test()) {
        const OffsetVector a(dim, rng);
        for (const KernelSet& set : vecdex::kernels::available()) {
            EXPECT_EQ(set.l2_squared(a.data(), a.data(), dim), 0.0F) << label(set, dim);
        }
    }
}

// On small integers every partial sum is exact (below 2^24), so every kernel must return the
// same bits whatever its summation order. This is why exact recall on SIFT survives SIMD.
TEST(Kernels, AllSetsAgreeExactlyOnSmallIntegers) {
    std::mt19937 rng(13);
    std::uniform_int_distribution<int> byte(0, 255);
    constexpr std::size_t kDim = 128;
    std::vector<float> a(kDim);
    std::vector<float> b(kDim);
    for (std::size_t i = 0; i < kDim; ++i) {
        a[i] = static_cast<float>(byte(rng));
        b[i] = static_cast<float>(byte(rng));
    }
    const KernelSet& scalar = vecdex::kernels::available().front();
    const float l2 = scalar.l2_squared(a.data(), b.data(), kDim);
    const float ip = scalar.inner_product(a.data(), b.data(), kDim);
    for (const KernelSet& set : vecdex::kernels::available()) {
        EXPECT_EQ(set.l2_squared(a.data(), b.data(), kDim), l2) << label(set, kDim);
        EXPECT_EQ(set.inner_product(a.data(), b.data(), kDim), ip) << label(set, kDim);
    }
}

TEST(Kernels, NamesAreStable) {
    EXPECT_EQ(vecdex::kernels::name(Isa::Scalar), "scalar");
    EXPECT_EQ(vecdex::kernels::name(Isa::Neon), "neon");
    EXPECT_EQ(vecdex::kernels::name(Isa::Avx2), "avx2");
}

}  // namespace
