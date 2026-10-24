#pragma once

// Each benchmark file registers its benchmarks at runtime, because what exists depends on the
// machine: which kernel sets the CPU supports, and whether the dataset was downloaded.

namespace vecdex::bench {

void register_distance_benchmarks();
void register_brute_force_benchmarks();

}  // namespace vecdex::bench
