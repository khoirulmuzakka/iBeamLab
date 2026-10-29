@echo off
setlocal
cd /d "%~dp0"

set "BUILD_OPTION=%~1"
if not defined BUILD_OPTION set "BUILD_OPTION=release"
set "CONFIG="

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

rem Separate configurations so Debug and Release never replace each other's outputs.
set "IBEAMLAB_BUILD_DIR=build\local\%CONFIG%"
set "ONNX_ENABLED=OFF"
if defined ONNXRUNTIME_ROOT set "ONNX_ENABLED=ON"

rem Pin CMake to the interpreter on PATH and check the isolated test environment.
set "IBEAMLAB_PYTHON_EXECUTABLE="
for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)"') do set "IBEAMLAB_PYTHON_EXECUTABLE=%%P"
if not defined IBEAMLAB_PYTHON_EXECUTABLE (
    echo Activate the Python environment containing numpy, pybind11, pywin32 and tqdm.
    exit /b 2
)
"%IBEAMLAB_PYTHON_EXECUTABLE%" -I -c "import numpy, pybind11, pythoncom, tqdm"
if errorlevel 1 (
    echo Missing dependencies in %IBEAMLAB_PYTHON_EXECUTABLE%.
    echo Install numpy, pybind11, pywin32 and tqdm into this environment, not the user site.
    echo Or activate your DL environment before running compile.bat.
    exit /b 2
)
if "%ONNX_ENABLED%"=="ON" (
    "%IBEAMLAB_PYTHON_EXECUTABLE%" -I -c "import torch, onnx"
    if errorlevel 1 (
        echo ONNX export tests require torch and onnx in this environment.
        exit /b 2
    )
)

echo Building %CONFIG% in %IBEAMLAB_BUILD_DIR%
echo Python: %IBEAMLAB_PYTHON_EXECUTABLE%
cmake -S . -B "%IBEAMLAB_BUILD_DIR%" -DCMAKE_BUILD_TYPE=%CONFIG% "-DPython_EXECUTABLE=%IBEAMLAB_PYTHON_EXECUTABLE%" -DBUILD_SHARED_LIBS=ON -DIBEAMLAB_BUILD_TESTS=ON -DIBEAMLAB_BUILD_PYTHON=ON -DIBEAMLAB_BUILD_EXAMPLES=ON -DIBEAMLAB_BUILD_BENCHMARKS=ON -DIBEAMLAB_ENABLE_SIMNRA=ON -DIBEAMLAB_REQUIRE_SIMNRA_TEST_RUNTIME=ON -DIBEAMLAB_ENABLE_ONNX=%ONNX_ENABLED% "-DONNXRUNTIME_ROOT=%ONNXRUNTIME_ROOT%"
if errorlevel 1 exit /b %errorlevel%
cmake --build "%IBEAMLAB_BUILD_DIR%" --config "%CONFIG%" --parallel
if errorlevel 1 exit /b %errorlevel%
ctest --test-dir "%IBEAMLAB_BUILD_DIR%" -C "%CONFIG%" --no-tests=error --output-on-failure
if errorlevel 1 exit /b %errorlevel%
echo iBeamLab build completed successfully.
echo C++ import library: %IBEAMLAB_BUILD_DIR%\lib\ibeamlab.lib
echo Python extension and runtime DLLs: %IBEAMLAB_BUILD_DIR%\bin\
echo ONNX inference enabled: %ONNX_ENABLED%
endlocal
exit /b 0
