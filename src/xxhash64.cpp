#include "xxhash64.hpp"

#include <algorithm>
#include <bit>
#include <cstring>

namespace vecdex::detail {
namespace {

// The spec reads every lane as a little-endian integer; on a little-endian host that is a plain
// load. memcpy rather than a pointer cast: the input has no alignment guarantee.
static_assert(std::endian::native == std::endian::little,
              "XXH64 lanes are little-endian; a big-endian host would need byte swapping");

constexpr std::uint64_t kPrime1 = 0x9E3779B185EBCA87ULL;
constexpr std::uint64_t kPrime2 = 0xC2B2AE3D27D4EB4FULL;
constexpr std::uint64_t kPrime3 = 0x165667B19E3779F9ULL;
constexpr std::uint64_t kPrime4 = 0x85EBCA77C2B2AE63ULL;
constexpr std::uint64_t kPrime5 = 0x27D4EB2F165667C5ULL;

std::uint64_t read64(const std::byte* p) noexcept {
    std::uint64_t value = 0;
    std::memcpy(&value, p, sizeof value);
    return value;
}

std::uint32_t read32(const std::byte* p) noexcept {
    std::uint32_t value = 0;
    std::memcpy(&value, p, sizeof value);
    return value;
}

constexpr std::uint64_t round_lane(std::uint64_t acc, std::uint64_t lane) noexcept {
    acc += lane * kPrime2;
    acc = std::rotl(acc, 31);
    return acc * kPrime1;
}

constexpr std::uint64_t merge_accumulator(std::uint64_t acc, std::uint64_t lane_acc) noexcept {
    acc ^= round_lane(0, lane_acc);
    return acc * kPrime1 + kPrime4;
}

}  // namespace

Xxh64::Xxh64(std::uint64_t seed) noexcept
    : seed_(seed), acc_{seed + kPrime1 + kPrime2, seed + kPrime2, seed, seed - kPrime1} {}

void Xxh64::consume(const std::byte* stripe) noexcept {
    for (std::size_t lane = 0; lane < acc_.size(); ++lane) {
        acc_[lane] = round_lane(acc_[lane], read64(stripe + 8 * lane));
    }
}

void Xxh64::update(std::span<const std::byte> bytes) noexcept {
    length_ += bytes.size();
    if (pending_size_ > 0) {  // first complete the stripe a previous call started
        const std::size_t take = std::min(kStripe - pending_size_, bytes.size());
        std::ranges::copy(bytes.first(take), pending_.begin() + pending_size_);
        pending_size_ += take;
        bytes = bytes.subspan(take);
        if (pending_size_ < kStripe) {
            return;
        }
        consume(pending_.data());
        pending_size_ = 0;
    }
    for (; bytes.size() >= kStripe; bytes = bytes.subspan(kStripe)) {
        consume(bytes.data());
    }
    std::ranges::copy(bytes, pending_.begin());
    pending_size_ = bytes.size();
}

std::uint64_t Xxh64::digest() const noexcept {
    std::uint64_t acc = 0;
    if (length_ >= kStripe) {  // the four accumulators converge into one
        acc = std::rotl(acc_[0], 1) + std::rotl(acc_[1], 7) + std::rotl(acc_[2], 12) +
              std::rotl(acc_[3], 18);
        for (const std::uint64_t lane_acc : acc_) {
            acc = merge_accumulator(acc, lane_acc);
        }
    } else {  // too short for a single stripe: the accumulators were never used
        acc = seed_ + kPrime5;
    }
    acc += length_;

    // The last length % 32 bytes: 8, then 4, then 1 at a time.
    std::span<const std::byte> rest(pending_.data(), pending_size_);
    for (; rest.size() >= 8; rest = rest.subspan(8)) {
        acc ^= round_lane(0, read64(rest.data()));
        acc = std::rotl(acc, 27) * kPrime1 + kPrime4;
    }
    if (rest.size() >= 4) {
        acc ^= std::uint64_t{read32(rest.data())} * kPrime1;
        acc = std::rotl(acc, 23) * kPrime2 + kPrime3;
        rest = rest.subspan(4);
    }
    for (const std::byte b : rest) {
        acc ^= std::to_integer<std::uint64_t>(b) * kPrime5;
        acc = std::rotl(acc, 11) * kPrime1;
    }

    // Avalanche: every input bit affects every output bit.
    acc ^= acc >> 33U;
    acc *= kPrime2;
    acc ^= acc >> 29U;
    acc *= kPrime3;
    acc ^= acc >> 32U;
    return acc;
}

std::uint64_t xxh64(std::span<const std::byte> bytes, std::uint64_t seed) noexcept {
    Xxh64 hash(seed);
    hash.update(bytes);
    return hash.digest();
}

}  // namespace vecdex::detail
