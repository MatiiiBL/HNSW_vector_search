include(FetchContent)

# Pinned by version and checksum: the build downloads exactly this code, every time.
FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(vecdex_tests
    brute_force_test.cpp
    distance_test.cpp
    hnsw_io_test.cpp
    hnsw_parallel_test.cpp
    hnsw_test.cpp
    io_test.cpp
    kernels_test.cpp
    matrix_test.cpp
    recall_test.cpp
    siftsmall_test.cpp
    smoke_test.cpp
    xxhash64_test.cpp)

# Some tests reach internal headers: the checksum, to craft files that pass it.
target_include_directories(vecdex_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(vecdex_tests PRIVATE
    vecdex::vecdex
    vecdex_warnings
    GTest::gtest_main)

target_compile_definitions(vecdex_tests PRIVATE
    VECDEX_DATA_DIR="${PROJECT_SOURCE_DIR}/data"
    $<$<BOOL:${VECDEX_SANITIZE}>:VECDEX_SANITIZE=1>)

include(GoogleTest)
gtest_discover_tests(vecdex_tests)

# A libFuzzer target for HnswIndex::load (see tests/load_fuzzer.cpp for how to build and run it).
# The library gets coverage instrumentation too, so the fuzzer can steer by what load() does.
# That instrumentation needs the sanitizer runtime, hence the VECDEX_SANITIZE requirement.
if(VECDEX_BUILD_FUZZERS)
    if(NOT VECDEX_SANITIZE)
        message(FATAL_ERROR "VECDEX_BUILD_FUZZERS needs VECDEX_SANITIZE=ON")
    endif()
    target_compile_options(vecdex PRIVATE -fsanitize=fuzzer-no-link)
    add_executable(vecdex_load_fuzzer load_fuzzer.cpp)
    target_compile_options(vecdex_load_fuzzer PRIVATE -fsanitize=fuzzer)
    target_link_options(vecdex_load_fuzzer PRIVATE -fsanitize=fuzzer)
    target_include_directories(vecdex_load_fuzzer PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(vecdex_load_fuzzer PRIVATE vecdex::vecdex vecdex_warnings)
endif()
