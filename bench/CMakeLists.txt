include(FetchContent)

# Pinned by version and checksum, like GoogleTest.
FetchContent_Declare(benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.5.tar.gz
    URL_HASH SHA256=9631341c82bac4a288bef951f8b26b41f69021794184ece969f8473977eaa340
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
# Their -Werror would turn a new compiler's warnings in *their* code into our build failure.
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# hnswlib, the reference HNSW implementation, runs side by side with vecdex in vecdex_eval.
# Header-only. SYSTEM marks its headers as system headers, so our warning flags skip them.
FetchContent_Declare(hnswlib
    URL https://github.com/nmslib/hnswlib/archive/refs/tags/v0.8.0.tar.gz
    URL_HASH SHA256=cf61d7dd8dc9bfba7f4abe0ed26698e90ac9f49a0badb2e1b0f3ba89b72cf3bb
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM)
FetchContent_MakeAvailable(hnswlib)

add_executable(vecdex_bench
    brute_force_bench.cpp
    distance_bench.cpp
    main.cpp)

target_link_libraries(vecdex_bench PRIVATE
    vecdex::vecdex
    vecdex_warnings
    benchmark::benchmark)

target_compile_definitions(vecdex_bench PRIVATE
    VECDEX_DATA_DIR="${PROJECT_SOURCE_DIR}/data")

# Recall-vs-QPS sweeps on a full dataset, for vecdex and hnswlib alike.
add_executable(vecdex_eval hnsw_eval.cpp)
target_link_libraries(vecdex_eval PRIVATE vecdex::vecdex vecdex_warnings hnswlib::hnswlib)
