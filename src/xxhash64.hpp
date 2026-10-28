#pragma once

// Internal: XXH64, the 64-bit xxHash, as the checksum of saved index files.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace vecdex::detail {

// Streaming XXH64, implemented from the specification in the xxHash repository
// (doc/xxhash_spec.md). Feeding the input in any number of update() calls gives the same digest
// as one call over all of it. It runs at several bytes per cycle, so checksumming a file costs
// little next to reading it.
//
// A checksum against accidental corruption (truncated copies, flipped bits), not an
// authenticator: anyone who edits a file can recompute it, which is why load() validates the
// contents as well.
class Xxh64 {
public:
    explicit Xxh64(std::uint64_t seed = 0) noexcept;

    void update(std::span<const std::byte> bytes) noexcept;

    // The hash of everything passed to update() so far. Doesn't change the state.
    std::uint64_t digest() const noexcept;

private:
    static constexpr std::size_t kStripe = 32;  // bytes consumed per round of the 4 accumulators

    void consume(const std::byte* stripe) noexcept;

    std::uint64_t seed_;
    std::array<std::uint64_t, 4> acc_;
    std::array<std::byte, kStripe> pending_{};  // the start of a stripe not yet complete
    std::size_t pending_size_ = 0;
    std::uint64_t length_ = 0;  // total bytes hashed
};

// XXH64 of `bytes` in one call.
std::uint64_t xxh64(std::span<const std::byte> bytes, std::uint64_t seed = 0) noexcept;

}  // namespace vecdex::detail
