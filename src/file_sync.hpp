#pragma once

// Internal: flushing files to stable storage, which makes HnswIndex::save durable.

#include <filesystem>
#include <system_error>

namespace vecdex::detail {

// Flushes `file`'s data and metadata past every cache, the drive's included, so that they
// survive a power loss. It doesn't matter which descriptor or stream wrote the data: this opens
// the file again, read-only, to flush it. Returns the error, or an empty error_code on success.
std::error_code sync_file(const std::filesystem::path& file) noexcept;

// The same for a directory's entries, so that a file created in or renamed into `dir` keeps its
// name after a power loss.
std::error_code sync_directory(const std::filesystem::path& dir) noexcept;

}  // namespace vecdex::detail
