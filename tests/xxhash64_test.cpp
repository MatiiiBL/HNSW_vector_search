#include "xxhash64.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

using vecdex::detail::xxh64;
using vecdex::detail::Xxh64;

constexpr std::uint64_t kPrime32 = 2654435761U;

// The input of xxHash's own sanity check (XSUM_fillTestBuffer in cli/xsum_sanity_check.c), so the
// expected hashes can be copied from there unchanged.
std::vector<std::byte> sanity_buffer(std::size_t size) {
    constexpr std::uint64_t kPrime64 = 11400714785074694797ULL;
    std::vector<std::byte> buffer(size);
    std::uint64_t generator = kPrime32;
    for (std::byte& b : buffer) {
        b = static_cast<std::byte>(generator >> 56U);
        generator *= kPrime64;
    }
    return buffer;
}

// XSUM_XXH64_testdata from xxHash v0.8.3. The lengths cover the short path (< 32 bytes), the
// stripe path, and every kind of tail (8-, 4- and 1-byte steps); the seeds cover initialization.
TEST(Xxh64, MatchesTheReferenceVectors) {
    struct Case {
        std::size_t length;
        std::uint64_t seed;
        std::uint64_t hash;
    };
    constexpr std::array<Case, 9> kCases{{
        {0, 0, 0xEF46DB3751D8E999ULL},
        {0, kPrime32, 0xAC75FDA2929B17EFULL},
        {1, 0, 0xE934A84ADB052768ULL},
        {1, kPrime32, 0x5014607643A9B4C3ULL},
        {4, 0, 0x9136A0DCA57457EEULL},
        {14, 0, 0x8282DCC4994E35C8ULL},
        {14, kPrime32, 0xC3BD6BF63DEB6DF0ULL},
        {222, 0, 0xB641AE8CB691C174ULL},
        {222, kPrime32, 0x20CB8AB7AE10C14AULL},
    }};
    const std::vector<std::byte> buffer = sanity_buffer(222);
    for (const Case& c : kCases) {
        EXPECT_EQ(xxh64(std::span(buffer).first(c.length), c.seed), c.hash)
            << "length " << c.length << ", seed " << c.seed;
    }
}

// Longer inputs, checked against an independent implementation: a zstd frame's content checksum
// is the low 32 bits of XXH64 (seed 0) of the content. Expected values from
// `zstd -c --check buffer | tail -c 4` (zstd 1.5.7, 2026-10-27).
TEST(Xxh64, LowBitsMatchZstdOnLongInputs) {
    struct Case {
        std::size_t length;
        std::uint32_t low_bits;
    };
    constexpr std::array<Case, 3> kCases{{
        {4096, 0x85F4E70BU},
        {100'003, 0x1682A665U},
        {1'048'583, 0x4B6FAE5CU},
    }};
    for (const Case& c : kCases) {
        EXPECT_EQ(static_cast<std::uint32_t>(xxh64(sanity_buffer(c.length))), c.low_bits)
            << "length " << c.length;
    }
}

TEST(Xxh64, StreamingMatchesOneShotForEverySplitPoint) {
    const std::vector<std::byte> buffer = sanity_buffer(300);
    const std::uint64_t expected = xxh64(buffer);
    for (std::size_t split = 0; split <= buffer.size(); ++split) {
        Xxh64 hash;
        hash.update(std::span(buffer).first(split));
        hash.update(std::span(buffer).subspan(split));
        EXPECT_EQ(hash.digest(), expected) << "split at " << split;
    }
}

// Pieces of 1 to 40 bytes, so updates start and end at every offset within a 32-byte stripe.
TEST(Xxh64, StreamingInUnevenPiecesMatchesOneShot) {
    const std::vector<std::byte> buffer = sanity_buffer(5000);
    Xxh64 hash;
    std::span<const std::byte> rest(buffer);
    for (std::size_t piece = 1; !rest.empty(); piece = piece % 40 + 1) {
        const std::span<const std::byte> part = rest.first(std::min(piece, rest.size()));
        hash.update(part);
        rest = rest.subspan(part.size());
    }
    EXPECT_EQ(hash.digest(), xxh64(buffer));
}

TEST(Xxh64, DigestLeavesTheStateUnchanged) {
    const std::vector<std::byte> buffer = sanity_buffer(100);
    Xxh64 hash;
    hash.update(std::span(buffer).first(50));
    EXPECT_EQ(hash.digest(), hash.digest());
    EXPECT_EQ(hash.digest(), xxh64(std::span(buffer).first(50)));
    hash.update(std::span(buffer).subspan(50));
    EXPECT_EQ(hash.digest(), xxh64(buffer));
}

}  // namespace
