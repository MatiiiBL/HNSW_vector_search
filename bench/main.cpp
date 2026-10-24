// Run a Release build only (never a sanitizer build), e.g.:
//   build/release/bench/vecdex_bench --benchmark_repetitions=5 \
//       --benchmark_report_aggregates_only=true
// VECDEX_ISA=scalar makes the indexes use the scalar kernels, to time them against SIMD.

#include <benchmark/benchmark.h>

#include "benchmarks.hpp"

int main(int argc, char** argv) {
    vecdex::bench::register_distance_benchmarks();
    vecdex::bench::register_brute_force_benchmarks();
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
