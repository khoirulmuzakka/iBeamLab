if(NOT DEFINED IBEAMLAB_BUILD_DIR OR NOT DEFINED IBEAMLAB_SOURCE_DIR)
    message(FATAL_ERROR "installed consumer test requires build and source directories")
endif()

set(test_root "${IBEAMLAB_BUILD_DIR}/installed-consumer-test")
set(install_prefix "${test_root}/install")
set(consumer_build "${test_root}/build")
set(package_path "${test_root}/runtime-package")
file(REMOVE_RECURSE "${test_root}")

set(config_args)
set(build_config_args)
if(DEFINED IBEAMLAB_CONFIG AND NOT IBEAMLAB_CONFIG STREQUAL "")
    list(APPEND config_args --config "${IBEAMLAB_CONFIG}")
    list(APPEND build_config_args --config "${IBEAMLAB_CONFIG}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${IBEAMLAB_BUILD_DIR}"
            --prefix "${install_prefix}" ${config_args}
    RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "failed to install iBeamLab for the external consumer test")
endif()

set(configure_command
    "${CMAKE_COMMAND}"
    -S "${IBEAMLAB_SOURCE_DIR}/native/tests/consumer"
    -B "${consumer_build}"
    "-DCMAKE_PREFIX_PATH=${install_prefix}")
if(DEFINED IBEAMLAB_GENERATOR AND NOT IBEAMLAB_GENERATOR STREQUAL "")
    list(APPEND configure_command -G "${IBEAMLAB_GENERATOR}")
endif()
if(DEFINED IBEAMLAB_GENERATOR_PLATFORM AND NOT IBEAMLAB_GENERATOR_PLATFORM STREQUAL "")
    list(APPEND configure_command -A "${IBEAMLAB_GENERATOR_PLATFORM}")
endif()
if(DEFINED IBEAMLAB_GENERATOR_TOOLSET AND NOT IBEAMLAB_GENERATOR_TOOLSET STREQUAL "")
    list(APPEND configure_command -T "${IBEAMLAB_GENERATOR_TOOLSET}")
endif()
execute_process(COMMAND ${configure_command} RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "failed to configure the external iBeamLab consumer")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${consumer_build}" ${build_config_args}
    RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "failed to build the external iBeamLab consumer")
endif()

if(WIN32)
    if(DEFINED IBEAMLAB_CONFIG AND NOT IBEAMLAB_CONFIG STREQUAL "")
        set(consumer_executable "${consumer_build}/${IBEAMLAB_CONFIG}/ibeamlab_consumer.exe")
    else()
        set(consumer_executable "${consumer_build}/ibeamlab_consumer.exe")
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env "PATH=${install_prefix}/bin;$ENV{PATH}"
                "${consumer_executable}" "${IBEAMLAB_TEST_MODEL}" "${package_path}"
        RESULT_VARIABLE result)
else()
    set(consumer_executable "${consumer_build}/ibeamlab_consumer")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
                "LD_LIBRARY_PATH=${install_prefix}/lib:$ENV{LD_LIBRARY_PATH}"
                "${consumer_executable}" "${IBEAMLAB_TEST_MODEL}" "${package_path}"
        RESULT_VARIABLE result)
endif()
if(NOT result EQUAL 0)
    message(FATAL_ERROR "installed external iBeamLab consumer failed at runtime")
endif()
