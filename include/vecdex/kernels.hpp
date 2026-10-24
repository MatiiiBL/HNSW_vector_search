#pragma once

// Low-level distance kernels, one set per instruction set (ISA). The library picks the fastest
// set this CPU supports, once, at runtime. Tests and benchmarks use this header to reach every
// set directly; most code wants distance.hpp instead.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace vecdex::kernels {

// Every kernel takes two arrays of `dim` floats. No alignment is required.
using DistanceFn = float (*)(const float* a, const float* b, std::size_t dim) noexcept;

enum class Isa : std::uint8_t { Scalar, Neon, Avx2 };

// "scalar", "neon" or "avx2".
std::string_view name(Isa isa) noexcept;

struct KernelSet {
    Isa isa;
    DistanceFn l2_squared;     // ||a - b||^2
    DistanceFn inner_product;  // <a, b>
};

// The sets this build contains and this CPU can run: scalar first, fastest last.
std::span<const KernelSet> available() noexcept;

// The set the library uses, chosen once (thread-safely) on first use: the fastest available
// one, unless the environment variable VECDEX_ISA names another available set. An unknown or
// unavailable name is reported on stderr and ignored.
const KernelSet& active() noexcept;

}  // namespace vecdex::kernels
