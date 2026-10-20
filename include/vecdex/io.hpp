#pragma once

#include <cstdint>
#include <filesystem>

#include "vecdex/matrix.hpp"

namespace vecdex {

// Readers for the TEXMEX formats used by SIFT and ann-benchmarks. Each record is a
// little-endian int32 dimension d followed by d 4-byte values (float32 for .fvecs, int32
// for .ivecs). Every record in a file must have the same d.
//
// Nothing in the file is trusted: the row count comes from the file size, not a header, so
// a corrupted file can't trigger a huge allocation. Throws std::runtime_error on any I/O or
// format error, with the path in the message.
Matrix<float> read_fvecs(const std::filesystem::path& path);
Matrix<std::int32_t> read_ivecs(const std::filesystem::path& path);

}  // namespace vecdex
