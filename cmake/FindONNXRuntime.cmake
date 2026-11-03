# Locate an existing SDK. Dependency preparation belongs to build tools, not CMake.
if(NOT DEFINED IBEAMLAB_ONNX_PROVIDER)
    set(IBEAMLAB_ONNX_PROVIDER CPU)
endif()
set(_ort_roots "${ONNXRUNTIME_ROOT}" "$ENV{ONNXRUNTIME_ROOT}")
if(NOT "${_ort_roots}" STREQUAL "${_IBEAMLAB_ORT_SEARCH_ROOTS}")
    foreach(_entry ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY ONNXRUNTIME_DLL
                   ONNXRUNTIME_cuda_RUNTIME ONNXRUNTIME_shared_RUNTIME)
        unset(${_entry} CACHE)
    endforeach()
endif()
set(_IBEAMLAB_ORT_SEARCH_ROOTS "${_ort_roots}" CACHE INTERNAL "SDK search roots" FORCE)
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h HINTS ${_ort_roots}
    PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session
                  build/native/include buildTransitive/native/include)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime HINTS ${_ort_roots}
    PATH_SUFFIXES lib lib64 runtimes/win-x64/native)
set(_ort_required ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY)
set(ONNXRUNTIME_RUNTIME_FILES)
if(WIN32)
    find_file(ONNXRUNTIME_DLL onnxruntime.dll HINTS ${_ort_roots}
        PATH_SUFFIXES bin lib runtimes/win-x64/native)
    list(APPEND _ort_required ONNXRUNTIME_DLL)
    list(APPEND ONNXRUNTIME_RUNTIME_FILES "${ONNXRUNTIME_DLL}")
    set(_ort_prefix "")
    set(_ort_suffix .dll)
else()
    get_filename_component(_ort_libdir "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
    file(GLOB ONNXRUNTIME_RUNTIME_FILES "${_ort_libdir}/libonnxruntime.so*"
         "${_ort_libdir}/libonnxruntime*.dylib")
    set(_ort_prefix lib)
    set(_ort_suffix .so)
endif()
if(IBEAMLAB_ONNX_PROVIDER STREQUAL "CUDA")
    foreach(_provider cuda shared)
        find_file(ONNXRUNTIME_${_provider}_RUNTIME
            "${_ort_prefix}onnxruntime_providers_${_provider}${_ort_suffix}"
            HINTS ${_ort_roots} PATH_SUFFIXES bin lib lib64 runtimes/win-x64/native)
        list(APPEND _ort_required ONNXRUNTIME_${_provider}_RUNTIME)
        list(APPEND ONNXRUNTIME_RUNTIME_FILES "${ONNXRUNTIME_${_provider}_RUNTIME}")
    endforeach()
endif()
# Parse the declared API version for deployment manifests.
if(ONNXRUNTIME_INCLUDE_DIR)
    file(STRINGS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h" _ort_api
         REGEX "^#define ORT_API_VERSION [0-9]+")
    string(REGEX REPLACE ".*ORT_API_VERSION ([0-9]+).*" "\\1" ONNXRUNTIME_API_VERSION "${_ort_api}")
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime REQUIRED_VARS ${_ort_required}
    REASON_FAILURE_MESSAGE "Native inference requires an ONNX Runtime C/C++ SDK matching IBEAMLAB_ONNX_PROVIDER. Set ONNXRUNTIME_ROOT or run compile.bat.")
if(ONNXRuntime_FOUND AND NOT TARGET ONNXRuntime::Runtime)
    add_library(ONNXRuntime::Runtime UNKNOWN IMPORTED)
    set_target_properties(ONNXRuntime::Runtime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")
endif()
