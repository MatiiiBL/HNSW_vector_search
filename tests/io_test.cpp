#include "vecdex/io.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

// Appends the raw (little-endian on our targets) bytes of `value`.
template <typename T>
void put(std::vector<char>& bytes, T value) {
    std::array<char, sizeof(T)> raw{};
    std::memcpy(raw.data(), &value, sizeof(T));
    bytes.insert(bytes.end(), raw.begin(), raw.end());
}

// The bytes of a well-formed .fvecs/.ivecs file holding `rows`.
template <typename T>
std::vector<char> vecs_bytes(const std::vector<std::vector<T>>& rows) {
    std::vector<char> bytes;
    for (const auto& row : rows) {
        put(bytes, static_cast<std::int32_t>(row.size()));
        for (const T value : row) {
            put(bytes, value);
        }
    }
    return bytes;
}

// A file in the temp directory, named after the running test, deleted on scope exit.
class TempFile {
public:
    explicit TempFile(const std::vector<char>& bytes) : path_(unique_path()) {
        std::ofstream out(path_, std::ios::binary);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!out) {
            throw std::runtime_error("TempFile: cannot write " + path_.string());
        }
    }
    ~TempFile() {
        std::error_code ignored;
        fs::remove(path_, ignored);
    }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    TempFile(TempFile&&) = delete;
    TempFile& operator=(TempFile&&) = delete;

    const fs::path& path() const { return path_; }

private:
    static fs::path unique_path() {
        static int counter = 0;
        const auto* test = ::testing::UnitTest::GetInstance()->current_test_info();
        return fs::temp_directory_path() /
               (std::string("vecdex_") + test->test_suite_name() + "_" + test->name() + "_" +
                std::to_string(counter++) + ".vecs");
    }

    fs::path path_;
};

TEST(ReadFvecs, ReadsWellFormedFile) {
    const TempFile file(vecs_bytes<float>({{1.0F, 2.0F}, {3.5F, -4.0F}, {0.0F, 1e6F}}));
    const auto m = vecdex::read_fvecs(file.path());
    ASSERT_EQ(m.rows(), 3U);
    ASSERT_EQ(m.dim(), 2U);
    EXPECT_EQ(m.row(0)[0], 1.0F);
    EXPECT_EQ(m.row(1)[0], 3.5F);
    EXPECT_EQ(m.row(1)[1], -4.0F);
    EXPECT_EQ(m.row(2)[1], 1e6F);
}

TEST(ReadIvecs, ReadsWellFormedFile) {
    const TempFile file(vecs_bytes<std::int32_t>({{7, 1, 9}, {0, -3, 2147483647}}));
    const auto m = vecdex::read_ivecs(file.path());
    ASSERT_EQ(m.rows(), 2U);
    ASSERT_EQ(m.dim(), 3U);
    EXPECT_EQ(m.row(0)[2], 9);
    EXPECT_EQ(m.row(1)[1], -3);
    EXPECT_EQ(m.row(1)[2], 2147483647);
}

TEST(ReadFvecs, MissingFileThrows) {
    EXPECT_THROW(vecdex::read_fvecs(fs::temp_directory_path() / "vecdex_no_such_file.fvecs"),
                 std::runtime_error);
}

TEST(ReadFvecs, EmptyFileThrows) {
    const TempFile file({});
    EXPECT_THROW(vecdex::read_fvecs(file.path()), std::runtime_error);
}

TEST(ReadFvecs, NonPositiveDimensionThrows) {
    for (const std::int32_t dim : {0, -1}) {
        std::vector<char> bytes;
        put(bytes, dim);
        put(bytes, 1.0F);
        const TempFile file(bytes);
        EXPECT_THROW(vecdex::read_fvecs(file.path()), std::runtime_error) << "dim " << dim;
    }
}

TEST(ReadFvecs, TruncatedFileThrows) {
    std::vector<char> bytes = vecs_bytes<float>({{1.0F, 2.0F}, {3.0F, 4.0F}});
    bytes.resize(bytes.size() - 3);  // cut the last value short
    const TempFile file(bytes);
    EXPECT_THROW(vecdex::read_fvecs(file.path()), std::runtime_error);
}

// Same total size as a valid 2 x 2 file, so only the per-record check can catch it.
TEST(ReadFvecs, InconsistentRecordDimensionThrows) {
    std::vector<char> bytes = vecs_bytes<float>({{1.0F, 2.0F}});
    put(bytes, std::int32_t{1});
    put(bytes, 3.0F);
    put(bytes, 4.0F);
    const TempFile file(bytes);
    EXPECT_THROW(vecdex::read_fvecs(file.path()), std::runtime_error);
}

// A corrupted header must not be trusted as an allocation size (8 GiB here).
TEST(ReadFvecs, HugeDimensionInHeaderDoesNotAllocate) {
    std::vector<char> bytes;
    put(bytes, std::numeric_limits<std::int32_t>::max());
    put(bytes, 1.0F);
    const TempFile file(bytes);
    EXPECT_THROW(vecdex::read_fvecs(file.path()), std::runtime_error);
}

TEST(ReadFvecs, ErrorMessageNamesTheFile) {
    const TempFile file({});
    try {
        (void)vecdex::read_fvecs(file.path());
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(file.path().filename().string()), std::string::npos)
            << e.what();
    }
}

}  // namespace
