#include "vecdex/kernels.hpp"

#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "kernels_impl.hpp"

namespace vecdex::kernels {
namespace {

struct Registry {
    std::array<KernelSet, 3> sets{};
    std::size_t count = 0;

    void add(const KernelSet& set) noexcept {
        assert(count < sets.size());  // one slot per Isa
        sets[count++] = set;
    }
};

Registry detect() noexcept {
    Registry registry;
    registry.add({.isa = Isa::Scalar,
                  .l2_squared = &detail::l2_squared_scalar,
                  .inner_product = &detail::inner_product_scalar});
#ifdef VECDEX_HAVE_NEON
    registry.add({.isa = Isa::Neon,
                  .l2_squared = &detail::l2_squared_neon,
                  .inner_product = &detail::inner_product_neon});
#endif
#ifdef VECDEX_HAVE_AVX2
    if (detail::cpu_supports_avx2_fma()) {
        registry.add({.isa = Isa::Avx2,
                      .l2_squared = &detail::l2_squared_avx2,
                      .inner_product = &detail::inner_product_avx2});
    }
#endif
    return registry;
}

const KernelSet& choose() noexcept {
    const std::span<const KernelSet> sets = available();
    const KernelSet& fastest = sets.back();
    const char* requested = std::getenv("VECDEX_ISA");
    if (requested == nullptr || *requested == '\0') {
        return fastest;
    }
    for (const KernelSet& set : sets) {
        if (name(set.isa) == requested) {
            return set;
        }
    }
    const std::string_view fallback = name(fastest.isa);
    std::fprintf(stderr, "vecdex: VECDEX_ISA=%s is not available on this CPU; using %.*s\n",
                 requested, static_cast<int>(fallback.size()), fallback.data());
    return fastest;
}

}  // namespace

std::string_view name(Isa isa) noexcept {
    switch (isa) {
        case Isa::Scalar:
            return "scalar";
        case Isa::Neon:
            return "neon";
        case Isa::Avx2:
            return "avx2";
    }
    return "unknown";
}

// Function-local statics are initialized exactly once, even with concurrent first calls.
std::span<const KernelSet> available() noexcept {
    static const Registry registry = detect();
    return {registry.sets.data(), registry.count};
}

const KernelSet& active() noexcept {
    static const KernelSet& set = choose();
    return set;
}

}  // namespace vecdex::kernels
