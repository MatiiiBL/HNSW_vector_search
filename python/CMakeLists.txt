# The extension module vecdex._core. `pip install .` builds it through scikit-build-core, which
# supplies Python and the pybind11 pinned in pyproject.toml.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# pyproject.toml holds the package version; refuse to build a wheel whose version disagrees
# with the library's.
if(DEFINED SKBUILD_PROJECT_VERSION AND NOT SKBUILD_PROJECT_VERSION VERSION_EQUAL PROJECT_VERSION)
    message(FATAL_ERROR "pyproject.toml has version ${SKBUILD_PROJECT_VERSION}, but "
                        "CMakeLists.txt has ${PROJECT_VERSION}")
endif()

# The static library is linked into a shared module. On Linux that requires position-independent
# code. Hidden visibility keeps its symbols private to the module, as pybind11 does for the module
# itself; mixing the two visibilities makes Apple's linker warn.
set_target_properties(vecdex PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

pybind11_add_module(_core MODULE bindings.cpp)
# pybind11 and Python come in through imported targets, so their headers are system headers and
# our warning flags (vecdex_warnings) skip them.
target_link_libraries(_core PRIVATE vecdex::vecdex vecdex_warnings)
target_compile_definitions(_core PRIVATE VECDEX_VERSION="${PROJECT_VERSION}")

# Next to vecdex/__init__.py in the wheel.
install(TARGETS _core LIBRARY DESTINATION vecdex)
