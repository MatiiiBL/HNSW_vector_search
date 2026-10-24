// End-to-end exact search on siftsmall: 10,000 base vectors, dim 128, k = 10. Each iteration
// answers one query (cycling through the 100 queries), so items/s is queries per second.
// The index uses kernels::active(); run once with VECDEX_ISA=scalar to compare.

#include <benchmark/benchmark.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include "benchmarks.hpp"
#include "vecdex/brute_force.hpp"
#include "vecdex/io.hpp"
#include "vecdex/kernels.hpp"

namespace vecdex::bench {
namespace {

namespace fs = std::filesystem;

struct SiftSmall {
    BruteForceIndex index;
    Matrix<float> queries;
};

SiftSmall load(const fs::path& base, const fs::path& queries) {
    const Matrix<float> vectors = read_fvecs(base);
    BruteForceIndex index(vectors.dim());
    index.add(vectors);
    return {.index = std::move(index), .queries = read_fvecs(queries)};
}

void run_search(benchmark::State& state, const SiftSmall& data) {
    constexpr std::size_t kK = 10;
    std::size_t q = 0;
    for (auto _ : state) {
        auto result = data.index.search(data.queries.row(q), kK);
        benchmark::DoNotOptimize(result);
        q = (q + 1) % data.queries.rows();
    }
    state.SetItemsProcessed(state.iterations());
    state.SetLabel(std::string("kernels: ") + std::string(kernels::name(kernels::active().isa)));
}

}  // namespace

void register_brute_force_benchmarks() {
    const fs::path dir = fs::path(VECDEX_DATA_DIR) / "siftsmall";
    const fs::path base = dir / "siftsmall_base.fvecs";
    const fs::path queries = dir / "siftsmall_query.fvecs";
    if (!fs::exists(base) || !fs::exists(queries)) {
        benchmark::RegisterBenchmark("brute_force/siftsmall", [dir](benchmark::State& state) {
            state.SkipWithError("siftsmall not found in " + dir.string() + " (see README)");
        });
        return;
    }
    // Loaded and indexed once, shared by every run; the lambda keeps it alive.
    auto data = std::make_shared<const SiftSmall>(load(base, queries));
    benchmark::RegisterBenchmark("brute_force/siftsmall/k:10", [data](benchmark::State& state) {
        run_search(state, *data);
    })->Unit(benchmark::kMicrosecond);
}

}  // namespace vecdex::bench
