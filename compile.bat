@echo off
setlocal
cd /d "%~dp0"

set "BUILD_OPTION=%~1"
if not defined BUILD_OPTION set "BUILD_OPTION=release"

if /I "%BUILD_OPTION%"=="--release" set "CONFIG=Release"
if /I "%BUILD_OPTION%"=="release" set "CONFIG=Release"
if /I "%BUILD_OPTION%"=="--debug" set "CONFIG=Debug"
if /I "%BUILD_OPTION%"=="debug" set "CONFIG=Debug"
if /I "%BUILD_OPTION%"=="--relwithdebinfo" set "CONFIG=RelWithDebInfo"
if /I "%BUILD_OPTION%"=="relwithdebinfo" set "CONFIG=RelWithDebInfo"
if /I "%BUILD_OPTION%"=="--minsizerel" set "CONFIG=MinSizeRel"
if /I "%BUILD_OPTION%"=="minsizerel" set "CONFIG=MinSizeRel"

if not defined CONFIG (
    echo Unknown build configuration: %BUILD_OPTION%
    echo Usage: compile.bat [--release^|--debug^|--relwithdebinfo^|--minsizerel]
    exit /b 2
)

cmake -S . -B build -DBUILD_SHARED_LIBS=ON -DIBEAMLAB_BUILD_TESTS=ON -DIBEAMLAB_BUILD_PYTHON=ON -DIBEAMLAB_ENABLE_SIMNRA=ON -DIBEAMLAB_ENABLE_ONNX=ON
if errorlevel 1 exit /b %errorlevel%
cmake --build build --config "%CONFIG%"
if errorlevel 1 exit /b %errorlevel%
ctest --test-dir build -C "%CONFIG%" --output-on-failure
if errorlevel 1 exit /b %errorlevel%
if not exist lib mkdir lib
copy /Y "build\lib\ibeamlab.lib" "lib\" >nul

echo iBeamLab build completed successfully.
echo C++ import library: lib\ibeamlab.lib
echo Python extension and runtime DLLs: ibeamlab\
endlocal
