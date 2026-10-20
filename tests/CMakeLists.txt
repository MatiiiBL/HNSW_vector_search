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
    io_test.cpp
    matrix_test.cpp
    recall_test.cpp
    siftsmall_test.cpp
    smoke_test.cpp)

target_link_libraries(vecdex_tests PRIVATE
    vecdex::vecdex
    vecdex_warnings
    GTest::gtest_main)

target_compile_definitions(vecdex_tests PRIVATE
    VECDEX_DATA_DIR="${PROJECT_SOURCE_DIR}/data"
    $<$<BOOL:${VECDEX_SANITIZE}>:VECDEX_SANITIZE=1>)

include(GoogleTest)
gtest_discover_tests(vecdex_tests)
