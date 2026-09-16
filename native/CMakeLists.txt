set(IBEAMLAB_SOURCES
    src/sample.cpp
    src/sample_toml.cpp
    src/simulator.cpp
    src/simnra_simulator.cpp
    src/generation.cpp
    src/data_generator.cpp
    src/datasets.cpp
    src/preprocessing.cpp
    src/transforms.cpp
    src/fftconv.cpp
    src/model_package.cpp
    src/inference.cpp
    ${IBEAMLAB_MINIZ_SOURCES})

if(IBEAMLAB_ENABLE_SIMNRA AND WIN32)
    list(APPEND IBEAMLAB_SOURCES src/simnra/com.cpp src/simnra/simnra.cpp)
endif()

add_library(ibeamlab ${IBEAMLAB_SOURCES})
target_compile_features(ibeamlab PUBLIC cxx_std_20)
target_include_directories(ibeamlab
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        src
        src/simnra
        ${IBEAMLAB_MINIZ_INCLUDE_DIRS}
        $<TARGET_PROPERTY:tomlplusplus::tomlplusplus,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(ibeamlab PRIVATE
    IBEAMLAB_VERSION="${PROJECT_VERSION}"
    IBEAMLAB_BUILD_TYPE="$<CONFIG>"
    IBEAMLAB_PLATFORM="${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

if(MSVC)
    target_compile_options(ibeamlab PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(ibeamlab PRIVATE -Wall -Wextra -Wpedantic)
endif()

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ibeamlab PRIVATE OpenMP::OpenMP_CXX)
endif()

if(IBEAMLAB_ENABLE_SIMNRA AND WIN32)
    target_compile_definitions(ibeamlab PRIVATE IBEAMLAB_HAS_SIMNRA=1 _CRT_SECURE_NO_WARNINGS)
    target_link_libraries(ibeamlab PRIVATE ole32 oleaut32)
endif()

if(IBEAMLAB_HAS_ONNX)
    target_compile_definitions(ibeamlab PRIVATE IBEAMLAB_HAS_ONNX=1)
    target_link_libraries(ibeamlab PRIVATE ONNXRuntime::ONNXRuntime)
    if(ONNXRUNTIME_RUNTIME)
        add_custom_command(TARGET ibeamlab POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_RUNTIME}" $<TARGET_FILE_DIR:ibeamlab>)
    endif()
endif()

install(TARGETS ibeamlab EXPORT ibeamlab-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_executable(ibeamlab-generate-dummy tools/generate_dummy.cpp)
target_link_libraries(ibeamlab-generate-dummy PRIVATE ibeamlab)
install(TARGETS ibeamlab-generate-dummy RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(IBEAMLAB_BUILD_EXAMPLES)
    add_executable(ibeamlab-read-dataset examples/read_dataset.cpp)
    target_link_libraries(ibeamlab-read-dataset PRIVATE ibeamlab)
endif()

if(IBEAMLAB_BUILD_BENCHMARKS)
    add_executable(ibeamlab-benchmark benchmarks/benchmark.cpp)
    target_link_libraries(ibeamlab-benchmark PRIVATE ibeamlab)
endif()

if(IBEAMLAB_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _pybind11_dir OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    set(pybind11_DIR "${_pybind11_dir}")
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_ibeamlab_cpp
        python/module.cpp python/bind_sample.cpp python/bind_simulator.cpp
        python/bind_generation.cpp python/bind_datasets.cpp
        python/bind_preprocessing.cpp python/bind_model_inference.cpp)
    target_link_libraries(_ibeamlab_cpp PRIVATE ibeamlab)
    set_target_properties(_ibeamlab_cpp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/ibeamlab/$<0:>"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/ibeamlab/$<0:>")
    if(CMAKE_CONFIGURATION_TYPES)
        foreach(config ${CMAKE_CONFIGURATION_TYPES})
            string(TOUPPER "${config}" config_upper)
            set_target_properties(_ibeamlab_cpp PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${PROJECT_SOURCE_DIR}/ibeamlab"
                RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${PROJECT_SOURCE_DIR}/ibeamlab")
        endforeach()
    endif()
    add_custom_command(TARGET _ibeamlab_cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:ibeamlab> $<TARGET_FILE_DIR:_ibeamlab_cpp>)
    install(TARGETS _ibeamlab_cpp
        LIBRARY DESTINATION ibeamlab
        RUNTIME DESTINATION ibeamlab)
    install(FILES $<TARGET_FILE:ibeamlab> DESTINATION ibeamlab)
    if(IBEAMLAB_HAS_ONNX AND ONNXRUNTIME_RUNTIME)
        add_custom_command(TARGET _ibeamlab_cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_RUNTIME}" $<TARGET_FILE_DIR:_ibeamlab_cpp>)
        install(FILES "${ONNXRUNTIME_RUNTIME}" DESTINATION ${CMAKE_INSTALL_BINDIR})
        install(FILES "${ONNXRUNTIME_RUNTIME}" DESTINATION ibeamlab)
    endif()
endif()

if(IBEAMLAB_BUILD_TESTS)
    add_executable(ibeamlab-native-tests tests/native_tests.cpp)
    target_link_libraries(ibeamlab-native-tests PRIVATE ibeamlab)
    target_compile_definitions(ibeamlab-native-tests PRIVATE
        IBEAMLAB_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
    add_test(NAME ibeamlab-native-tests COMMAND ibeamlab-native-tests)
    if(IBEAMLAB_ENABLE_SIMNRA AND WIN32 AND DEFINED ENV{IBEAMLAB_SIMNRA_REFERENCE})
        add_executable(ibeamlab-simnra-integration tests/simnra_integration.cpp)
        target_link_libraries(ibeamlab-simnra-integration PRIVATE ibeamlab)
        add_test(NAME ibeamlab-simnra-integration COMMAND ibeamlab-simnra-integration
            "$ENV{IBEAMLAB_SIMNRA_REFERENCE}")
        if(IBEAMLAB_BUILD_PYTHON)
            add_test(NAME ibeamlab-simnra-python-integration
                COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/simnra_python_integration.py"
                    "$ENV{IBEAMLAB_SIMNRA_REFERENCE}")
        endif()
    endif()
    if(IBEAMLAB_BUILD_PYTHON)
        add_test(NAME ibeamlab-python-smoke
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/python_smoke.py")
    endif()
endif()
