function(ibeamlab_copy_runtime_dependencies target)
    if(TARGET ibeamlab::ibeamlab)
        get_target_property(_runtime_files ibeamlab::ibeamlab IBEAMLAB_RUNTIME_FILES)
        if(_runtime_files)
            add_custom_command(TARGET "${target}" POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    ${_runtime_files} "$<TARGET_FILE_DIR:${target}>"
                COMMAND_EXPAND_LISTS VERBATIM)
        endif()
    endif()
endfunction()
