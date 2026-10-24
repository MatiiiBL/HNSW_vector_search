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
