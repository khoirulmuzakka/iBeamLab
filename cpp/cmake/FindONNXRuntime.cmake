find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}" "$ENV{ONNXRUNTIME_ROOT}"
    PATH_SUFFIXES lib lib64 runtimes/win-x64/native)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime REQUIRED_VARS ONNXRUNTIME_LIBRARY)
if(ONNXRuntime_FOUND AND NOT TARGET ONNXRuntime::Runtime)
    add_library(ONNXRuntime::Runtime UNKNOWN IMPORTED)
    set_target_properties(ONNXRuntime::Runtime PROPERTIES IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}")
endif()
