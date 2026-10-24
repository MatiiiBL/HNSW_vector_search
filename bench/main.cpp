// Run a Release build only (never a sanitizer build). For recorded numbers, use
//   --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
// and report the medians. VECDEX_ISA=scalar makes the indexes use the scalar kernels, to time
// them against SIMD. (No line here may end in a backslash: that would continue the comment
// onto the next line, which GCC rejects under -Werror.)

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
