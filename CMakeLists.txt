cmake_minimum_required(VERSION 3.25)

project(vecdex
    VERSION 0.1.0
    DESCRIPTION "Approximate nearest-neighbor vector search (HNSW) in C++20"
    LANGUAGES CXX)

# -std=c++20 rather than the GNU dialect (-std=gnu++20).
set(CMAKE_CXX_EXTENSIONS OFF)

option(VECDEX_BUILD_TESTS "Build the vecdex test suite" ${PROJECT_IS_TOP_LEVEL})
option(VECDEX_SANITIZE "Instrument everything with AddressSanitizer + UBSan" OFF)
option(VECDEX_WARNINGS_AS_ERRORS "Treat compiler warnings in vecdex code as errors" ON)
option(VECDEX_BUILD_BENCHMARKS "Build the benchmarks (meaningful in Release only)" OFF)

# Sanitizers are directory-wide on purpose: every target, including GoogleTest, must be
# instrumented, and the flags must reach the linker too. This has to run before any target
# is created. -fno-sanitize-recover makes UBSan abort; by default it only prints and the
# test still passes.
if(VECDEX_SANITIZE)
    add_compile_options(
        -fsanitize=address,undefined
        -fno-sanitize-recover=all
        -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# Warnings are per-target: they apply to vecdex's own code only, never to dependencies.
add_library(vecdex_warnings INTERFACE)
target_compile_options(vecdex_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow
    $<$<BOOL:${VECDEX_WARNINGS_AS_ERRORS}>:-Werror>)

add_library(vecdex
    src/brute_force.cpp
    src/distance.cpp
    src/hnsw.cpp
    src/io.cpp
    src/kernels.cpp
    src/kernels_scalar.cpp
    src/recall.cpp
    src/vector_ops.cpp)
add_library(vecdex::vecdex ALIAS vecdex)

# SIMD kernels for the target architecture(s). Each file is also guarded by the preprocessor
# (src/kernels_impl.hpp), so an extra one would be harmless, just an empty object file.
set(vecdex_archs "${CMAKE_OSX_ARCHITECTURES}")
if(NOT vecdex_archs)
    set(vecdex_archs "${CMAKE_SYSTEM_PROCESSOR}")
endif()
if(vecdex_archs MATCHES "x86_64|AMD64|amd64")
    target_sources(vecdex PRIVATE src/kernels_avx2.cpp)
endif()
if(vecdex_archs MATCHES "arm64|aarch64|ARM64")
    target_sources(vecdex PRIVATE src/kernels_neon.cpp)
endif()

target_include_directories(vecdex PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
# PUBLIC: the headers use std::span, so anything that includes them needs C++20 too.
target_compile_features(vecdex PUBLIC cxx_std_20)
target_link_libraries(vecdex PRIVATE vecdex_warnings)

if(VECDEX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(VECDEX_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()
