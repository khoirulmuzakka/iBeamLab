cmake_minimum_required(VERSION 3.24)
project(iBeamLab VERSION 0.1.0 LANGUAGES C CXX)
option(BUILD_SHARED_LIBS "Build iBeamLab as shared libraries" ON)

option(IBEAMLAB_BUILD_TESTS "Build native tests" ON)
option(IBEAMLAB_BUILD_PYTHON "Build the Python extension" OFF)
option(IBEAMLAB_BUILD_EXAMPLES "Build native examples" ON)
option(IBEAMLAB_BUILD_BENCHMARKS "Build lightweight native benchmarks" OFF)
option(IBEAMLAB_ENABLE_SANITIZERS "Enable Address/Undefined sanitizers where supported" OFF)
option(IBEAMLAB_ENABLE_SIMNRA "Build the Windows SIMNRA backend" ON)
option(IBEAMLAB_ENABLE_ONNX "Build ONNX Runtime inference" ON)
option(IBEAMLAB_FETCH_ONNX "Fetch a pinned ONNX Runtime package when not installed" ON)
set(IBEAMLAB_ONNX_VERSION "1.22.1" CACHE STRING "ONNX Runtime version used by the fallback package")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
if(IBEAMLAB_ENABLE_SANITIZERS AND NOT MSVC)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
if(CMAKE_CONFIGURATION_TYPES)
    foreach(config ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER "${config}" config_upper)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/lib")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/lib")
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/bin")
    endforeach()
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FetchContent)
find_package(tomlplusplus CONFIG QUIET)
if(NOT tomlplusplus_FOUND)
    FetchContent_Declare(tomlplusplus
        GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
        GIT_TAG v3.4.0
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(tomlplusplus)
endif()
FetchContent_Declare(miniz
    GIT_REPOSITORY https://github.com/richgel999/miniz.git
    GIT_TAG 3.1.1
    GIT_SHALLOW TRUE)
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()
FetchContent_GetProperties(miniz)
if(NOT miniz_POPULATED)
    FetchContent_Populate(miniz)
endif()
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated/miniz")
file(WRITE "${CMAKE_BINARY_DIR}/generated/miniz/miniz_export.h" "#pragma once\n#define MINIZ_EXPORT\n")
set(IBEAMLAB_MINIZ_SOURCES
    "${miniz_SOURCE_DIR}/miniz.c" "${miniz_SOURCE_DIR}/miniz_zip.c"
    "${miniz_SOURCE_DIR}/miniz_tinfl.c" "${miniz_SOURCE_DIR}/miniz_tdef.c")
set(IBEAMLAB_MINIZ_INCLUDE_DIRS
    "${miniz_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/generated/miniz")

if(IBEAMLAB_ENABLE_ONNX)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS "${ONNXRuntime_ROOT}" ENV ONNXRuntime_ROOT PATH_SUFFIXES include build/native/include)
    find_library(ONNXRUNTIME_LIBRARY onnxruntime
        HINTS "${ONNXRuntime_ROOT}" ENV ONNXRuntime_ROOT PATH_SUFFIXES lib lib64 runtimes/win-x64/native)
    if((NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY) AND IBEAMLAB_FETCH_ONNX AND WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
        FetchContent_Declare(onnxruntime_package
            URL "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/${IBEAMLAB_ONNX_VERSION}/microsoft.ml.onnxruntime.${IBEAMLAB_ONNX_VERSION}.nupkg"
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_MakeAvailable(onnxruntime_package)
        set(ONNXRUNTIME_INCLUDE_DIR "${onnxruntime_package_SOURCE_DIR}/build/native/include")
        set(ONNXRUNTIME_LIBRARY "${onnxruntime_package_SOURCE_DIR}/runtimes/win-x64/native/onnxruntime.lib")
        set(ONNXRUNTIME_RUNTIME "${onnxruntime_package_SOURCE_DIR}/runtimes/win-x64/native/onnxruntime.dll")
    endif()
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
        add_library(ONNXRuntime::ONNXRuntime SHARED IMPORTED GLOBAL)
        set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
            IMPORTED_IMPLIB "${ONNXRUNTIME_LIBRARY}")
        if(ONNXRUNTIME_RUNTIME)
            set_property(TARGET ONNXRuntime::ONNXRuntime PROPERTY IMPORTED_LOCATION "${ONNXRUNTIME_RUNTIME}")
        endif()
        set(IBEAMLAB_HAS_ONNX TRUE)
    else()
        message(WARNING "ONNX Runtime was not found; inference will build as an unavailable stub")
    endif()
endif()
if(IBEAMLAB_BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(native)

configure_package_config_file(
    cmake/ibeamlab-config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/ibeamlab-config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ibeamlab"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ibeamlab-config-version.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)
install(EXPORT ibeamlab-targets
    FILE ibeamlab-targets.cmake
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ibeamlab"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ibeamlab-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ibeamlab-config-version.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ibeamlab"
)
