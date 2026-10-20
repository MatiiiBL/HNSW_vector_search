#include "vecdex/io.hpp"

#include <bit>
#include <format>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>

namespace vecdex {
namespace {

static_assert(std::endian::native == std::endian::little,
              "fvecs/ivecs are little-endian; a big-endian host would need byte swapping");

[[noreturn]] void fail(const std::filesystem::path& path, const std::string& what) {
    throw std::runtime_error(std::format("{}: {}", path.string(), what));
}

// Reads exactly `bytes` bytes into `dst`. Writing an object's bytes through char* is allowed;
// casting a byte buffer to float* instead would break strict aliasing and alignment rules.
void read_exact(std::istream& in, void* dst, std::size_t bytes, const std::filesystem::path& path) {
    in.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    if (in.gcount() != static_cast<std::streamsize>(bytes)) {
        fail(path, "unexpected end of file");
    }
}

template <typename T>
Matrix<T> read_vecs(const std::filesystem::path& path) {
    static_assert(sizeof(T) == 4, "fvecs/ivecs elements are 4 bytes");

    std::error_code ec;
    const std::uintmax_t file_size = std::filesystem::file_size(path, ec);
    if (ec) {
        fail(path, ec.message());
    }
    if (file_size == 0) {
        fail(path, "file is empty");
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail(path, "cannot open file");
    }

    std::int32_t dim = 0;
    read_exact(in, &dim, sizeof dim, path);
    if (dim <= 0) {
        fail(path, std::format("invalid dimension {} in the first record", dim));
    }

    // The row count comes from the file size, never from a header, so the allocation below
    // can't be larger than the file itself.
    const std::uintmax_t record_bytes =
        sizeof(std::int32_t) + static_cast<std::uintmax_t>(dim) * sizeof(T);
    if (file_size % record_bytes != 0) {
        fail(path, std::format("size {} is not a multiple of the record size {} (dimension {}); "
                               "the file is truncated or not in this format",
                               file_size, record_bytes, dim));
    }
    const auto rows = static_cast<std::size_t>(file_size / record_bytes);

    Matrix<T> matrix(rows, static_cast<std::size_t>(dim));
    in.seekg(0);
    for (std::size_t i = 0; i < rows; ++i) {
        std::int32_t record_dim = 0;
        read_exact(in, &record_dim, sizeof record_dim, path);
        if (record_dim != dim) {
            fail(path, std::format("record {} has dimension {}, expected {}", i, record_dim, dim));
        }
        const std::span<T> row = matrix.row(i);
        read_exact(in, row.data(), row.size_bytes(), path);
    }
    return matrix;
}

}  // namespace

Matrix<float> read_fvecs(const std::filesystem::path& path) {
    return read_vecs<float>(path);
}

Matrix<std::int32_t> read_ivecs(const std::filesystem::path& path) {
    return read_vecs<std::int32_t>(path);
}

}  // namespace vecdex
