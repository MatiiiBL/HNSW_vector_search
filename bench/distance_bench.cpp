// Distance kernel microbenchmarks: one call on two vectors that stay in L1 cache, so this
// measures arithmetic throughput. In a real scan the base vectors stream from memory, which is
// slower; brute_force_bench.cpp measures that end to end.

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "benchmarks.hpp"
#include "vecdex/kernels.hpp"

namespace vecdex::bench {
namespace {

using kernels::DistanceFn;

#ifdef __clang__
// The scalar loop with reassociation allowed in this one function: the effect of -ffast-math
// without its global side effects. Shows what the compiler's auto-vectorizer manages on its
// own once it may reorder the additions. (Clang only; GCC has no per-block equivalent.)
float l2_squared_reassociated(const float* a, const float* b, std::size_t dim) noexcept {
#pragma clang fp reassociate(on)
    float sum = 0.0F;
    for (std::size_t i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float inner_product_reassociated(const float* a, const float* b, std::size_t dim) noexcept {
#pragma clang fp reassociate(on)
    float sum = 0.0F;
    for (std::size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}
#endif

std::vector<float> random_vector(std::size_t dim, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    std::vector<float> v(dim);
    for (float& x : v) {
        x = uniform(rng);
    }
    return v;
}

void run_kernel(benchmark::State& state, DistanceFn kernel) {
    const auto dim = static_cast<std::size_t>(state.range(0));
    const std::vector<float> a = random_vector(dim, 1);
    const std::vector<float> b = random_vector(dim, 2);
    for (auto _ : state) {
        float result = kernel(a.data(), b.data(), dim);
        // Without this the compiler may delete the call, since its result is never used.
        benchmark::DoNotOptimize(result);
    }
    state.SetBytesProcessed(state.iterations() *
                            static_cast<std::int64_t>(2 * dim * sizeof(float)));
}

void register_kernel(const std::string& metric, const std::string& variant, DistanceFn kernel) {
    benchmark::RegisterBenchmark(metric + "/" + variant,
                                 [kernel](benchmark::State& state) { run_kernel(state, kernel); })
        ->ArgName("dim")
        ->Arg(96)
        ->Arg(128)
        ->Arg(384)
        ->Arg(768);
}

}  // namespace

void register_distance_benchmarks() {
    for (const kernels::KernelSet& set : kernels::available()) {
        const std::string isa(kernels::name(set.isa));
        register_kernel("l2_squared", isa, set.l2_squared);
        register_kernel("inner_product", isa, set.inner_product);
    }
#ifdef __clang__
    register_kernel("l2_squared", "scalar_reassociated", &l2_squared_reassociated);
    register_kernel("inner_product", "scalar_reassociated", &inner_product_reassociated);
#endif
}

}  // namespace vecdex::bench
