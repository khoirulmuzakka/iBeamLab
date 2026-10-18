set(IBEAMLAB_ONNX_PROVIDER "CPU" CACHE STRING "ONNX Runtime SDK capability: CPU or CUDA")
set_property(CACHE IBEAMLAB_ONNX_PROVIDER PROPERTY STRINGS CPU CUDA)
set(IBEAMLAB_ONNX_CUDA_VERSION "1.27.1" CACHE STRING "CUDA 13 ONNX Runtime GPU package version")
set(ONNXRuntime_ROOT "" CACHE PATH "Root of a matching ONNX Runtime SDK")
set(IBEAMLAB_CUDA_RUNTIME_DIRS "" CACHE STRING "Optional directories containing CUDA/cuDNN runtime DLLs to deploy")

set(IBEAMLAB_HAS_CUDA FALSE)

if(NOT IBEAMLAB_ONNX_PROVIDER MATCHES "^(CPU|CUDA)$")
    message(FATAL_ERROR "IBEAMLAB_ONNX_PROVIDER must be CPU or CUDA")
endif()

# This helper works both after FetchContent and from an installed package.
include("${CMAKE_CURRENT_LIST_DIR}/ibeamlab-runtime.cmake")
if(NOT IBEAMLAB_ENABLE_ONNX)
    return()
endif()

set(_ort_root "${ONNXRuntime_ROOT}")
if(NOT _ort_root AND DEFINED ENV{ONNXRuntime_ROOT})
    set(_ort_root "$ENV{ONNXRuntime_ROOT}")
endif()
if(NOT _ort_root AND IBEAMLAB_FETCH_ONNX AND WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    if(IBEAMLAB_ONNX_PROVIDER STREQUAL "CUDA")
        set(_ort_package microsoft.ml.onnxruntime.gpu.windows)
        set(_ort_version "${IBEAMLAB_ONNX_CUDA_VERSION}")
    else()
        set(_ort_package microsoft.ml.onnxruntime)
        set(_ort_version "${IBEAMLAB_ONNX_VERSION}")
    endif()
    FetchContent_Declare(onnxruntime_package
        URL "https://api.nuget.org/v3-flatcontainer/${_ort_package}/${_ort_version}/${_ort_package}.${_ort_version}.nupkg"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(onnxruntime_package)
    set(_ort_root "${onnxruntime_package_SOURCE_DIR}")
endif()

# Ignore cached paths when switching SDK or provider. Explicit roots never mix
# headers and libraries from a different installation.
unset(ONNXRUNTIME_INCLUDE_DIR CACHE)
unset(ONNXRUNTIME_LIBRARY CACHE)
unset(ONNXRUNTIME_RUNTIME CACHE)
set(_ort_search)
if(_ort_root)
    list(APPEND _ort_search PATHS "${_ort_root}" NO_DEFAULT_PATH)
endif()
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h ${_ort_search}
    PATH_SUFFIXES include include/onnxruntime/core/session build/native/include buildTransitive/native/include)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime ${_ort_search}
    PATH_SUFFIXES lib lib64 runtimes/win-x64/native)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "Cannot find a complete ONNX Runtime SDK. Set ONNXRuntime_ROOT or enable IBEAMLAB_FETCH_ONNX on Windows x64.")
endif()
get_filename_component(_ort_lib_dir "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
if(WIN32)
    find_file(ONNXRUNTIME_RUNTIME onnxruntime.dll
        PATHS "${_ort_lib_dir}" "${_ort_root}/bin" NO_DEFAULT_PATH)
    if(NOT ONNXRUNTIME_RUNTIME)
        message(FATAL_ERROR "Missing onnxruntime.dll in selected SDK: ${_ort_root}")
    endif()
else()
    set(ONNXRUNTIME_RUNTIME "${ONNXRUNTIME_LIBRARY}")
endif()
get_filename_component(_ort_runtime_dir "${ONNXRUNTIME_RUNTIME}" DIRECTORY)
set(ONNXRUNTIME_RUNTIME_FILES "${ONNXRUNTIME_RUNTIME}")
file(GLOB _ort_providers "${_ort_runtime_dir}/onnxruntime_providers_*.dll"
    "${_ort_runtime_dir}/libonnxruntime_providers_*.so*")
list(APPEND ONNXRUNTIME_RUNTIME_FILES ${_ort_providers})
if(IBEAMLAB_ONNX_PROVIDER STREQUAL "CUDA")
    if(NOT EXISTS "${_ort_runtime_dir}/onnxruntime_providers_cuda.dll" AND
       NOT EXISTS "${_ort_runtime_dir}/libonnxruntime_providers_cuda.so")
        message(FATAL_ERROR "CUDA requested but selected SDK has no CUDA provider: ${_ort_runtime_dir}")
    endif()
    set(IBEAMLAB_HAS_CUDA TRUE)
endif()
foreach(_dir IN LISTS IBEAMLAB_CUDA_RUNTIME_DIRS)
    if(NOT IS_DIRECTORY "${_dir}")
        message(FATAL_ERROR "CUDA runtime directory does not exist: ${_dir}")
    endif()
    file(GLOB _cuda_dlls "${_dir}/cud*.dll" "${_dir}/cublas*.dll"
        "${_dir}/cufft*.dll" "${_dir}/curand*.dll" "${_dir}/cus*.dll"
        "${_dir}/nvrtc*.dll" "${_dir}/nvJitLink*.dll")
    list(APPEND ONNXRUNTIME_RUNTIME_FILES ${_cuda_dlls})
endforeach()
list(REMOVE_DUPLICATES ONNXRUNTIME_RUNTIME_FILES)
if(NOT TARGET ONNXRuntime::ONNXRuntime)
    add_library(ONNXRuntime::ONNXRuntime SHARED IMPORTED GLOBAL)
    set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
        IMPORTED_LOCATION "${ONNXRUNTIME_RUNTIME}")
    if(WIN32)
        set_property(TARGET ONNXRuntime::ONNXRuntime PROPERTY IMPORTED_IMPLIB "${ONNXRUNTIME_LIBRARY}")
    endif()
else()
    message(FATAL_ERROR "An ONNXRuntime::ONNXRuntime target already exists; configure one SDK owner to avoid mixing runtimes.")
endif()
set(IBEAMLAB_HAS_ONNX TRUE)
message(STATUS "iBeamLab ONNX SDK: ${IBEAMLAB_ONNX_PROVIDER}; ${ONNXRUNTIME_LIBRARY}")


