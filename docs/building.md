# Building and installation

All builds start at the repository root. `CMakeLists.txt` defines project options
once and delegates library, extension and test targets to `cpp/CMakeLists.txt`.
Python packaging uses the same root project through `pyproject.toml`.
Use a fresh build directory for the unified layout, such as `build/local`.

## C++ library

Requirements: CMake 3.24+ and a C++20 compiler. The default build is static,
enables C++ tests, and requires no external simulator, ONNX SDK, Python or downloads.

```console
cmake -S . -B build/local -DCMAKE_BUILD_TYPE=Release
cmake --build build/local --config Release --parallel
ctest --test-dir build/local -C Release --output-on-failure
cmake --install build/local --config Release --prefix /path/to/install
```

Executables, shared libraries and the Python extension appear in the build
directory's `bin/`; static and import libraries appear in `lib/`.

Installed C++ consumers use:

```cmake
find_package(ibeamlab CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE ibeamlab::ibeamlab)
```

Pass `CMAKE_PREFIX_PATH` to locate the installation. On Windows, put the
installed `bin/` on the application's runtime search path for shared/ONNX builds.

## Optional features

| CMake option | Default | Purpose |
| --- | --- | --- |
| `BUILD_SHARED_LIBS` | OFF | Build a shared C++ library |
| `IBEAMLAB_BUILD_TESTS` | ON | Register C++ and enabled Python tests |
| `IBEAMLAB_CORE_BUILD_TESTS` | ON | Include core domain tests |
| `IBEAMLAB_BUILD_PYTHON` | OFF | Build the Python extension |
| `IBEAMLAB_BUILD_EXAMPLES` | OFF | Build the C++ generation example |
| `IBEAMLAB_BUILD_BENCHMARKS` | OFF | Build processing/inference allocation checks |
| `IBEAMLAB_ENABLE_SIMNRA` | OFF | Compile the Windows COM backend |
| `IBEAMLAB_ENABLE_ONNX` | OFF | Enable ONNX Runtime inference |
| `IBEAMLAB_ONNX_PROVIDER` | CPU | SDK capabilities: CPU or CUDA; CUDA builds can run either provider |
| `IBEAMLAB_SANITIZERS` | OFF | GCC/Clang address and undefined-behavior sanitizers |
| `IBEAMLAB_REQUIRE_SIMNRA_TEST_RUNTIME` | OFF | Fail licensed tests if SIMNRA is unavailable |
| `IBEAMLAB_CHECK_SAVED_SIMNRA_SPECTRA` | OFF | Also require stored XNRA curves to match fresh calculations |

SIMNRA requires Windows and a local SIMNRA installation. Its worker COM
apartments default to MTA (`multithreaded_apartment=true`). See [simnra.md](simnra.md).

ONNX inference requires a matching ONNX Runtime C/C++ SDK. Python's
`onnxruntime` package is not a substitute for that SDK. Set `ONNXRUNTIME_ROOT`:

```console
cmake -S . -B build/onnx -DIBEAMLAB_ENABLE_ONNX=ON -DONNXRUNTIME_ROOT=/path/to/sdk
cmake --build build/onnx --config Release --parallel
```

Windows builds copy the matching `onnxruntime.dll` beside build executables.
Redistribute runtime dependencies alongside consuming applications. The
installed-consumer test checks C++ package discovery and linking.

For CUDA, provide a GPU ONNX Runtime SDK (with CUDA and shared provider
libraries) and configure a separate build:

```console
cmake -S . -B build/onnx-cuda -DIBEAMLAB_ENABLE_ONNX=ON -DIBEAMLAB_ONNX_PROVIDER=CUDA -DONNXRUNTIME_ROOT=/path/to/gpu-sdk -DIBEAMLAB_BUILD_PYTHON=ON
cmake --build build/onnx-cuda --config Release --parallel
```

No CUDA compiler is needed for iBeamLab itself. Matching CUDA/cuDNN libraries
and a supported NVIDIA driver must be discoverable by the application. On
Windows, put their DLL directories on the process PATH before launching Python
or an external application. The SDK's `onnxruntime.dll`,
`onnxruntime_providers_cuda.dll` and `onnxruntime_providers_shared.dll` are copied
and installed together. CUDA/cuDNN libraries are supplied by the deployment
environment; see the official [CUDA runtime requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).
CUDA configuration supports the V2 provider API, including `use_tf32` (ORT 1.18+).

`compile.bat` uses `ONNXRUNTIME_ROOT` and `IBEAMLAB_ONNX_PROVIDER` environment
variables. Set the latter to `CUDA` for a GPU SDK; CPU remains the default.

## Python installation

Requirements: Python 3.10+, NumPy, CMake and a C++20 compiler. Pip installs the
scikit-build-core/pybind11 build dependencies in its isolated build environment.

```console
python -m pip install .
python -m pip install ".[training]"
```

The `training` extra adds PyTorch, ONNX and Python ONNX Runtime. The `examples` extra adds TOML
support on Python 3.10; Python 3.11+ includes it. Default wheels enable Python
bindings and a shared core, disable tests/examples, and leave ONNX/SIMNRA off.

Enable inference in a wheel build with explicit SDK settings:

```console
python -m pip install . --config-settings "cmake.define.IBEAMLAB_ENABLE_ONNX=ON" --config-settings "cmake.define.ONNXRUNTIME_ROOT=/path/to/sdk"
```

On Windows, add `--config-settings "cmake.define.IBEAMLAB_ENABLE_SIMNRA=ON"`
to compile the SIMNRA adapter.

For source-tree Python use, configure bindings explicitly:

```console
python -m pip install numpy pybind11
cmake -S . -B build/python -DIBEAMLAB_BUILD_PYTHON=ON -DPython_EXECUTABLE=/path/to/python
cmake --build build/python --config Release --parallel
```

Add `build/python/bin` to `PYTHONPATH`, or pass that directory as
`--extension-dir` to the data-generation example. The source wrapper loads the
build-tree extension lazily. See [python.md](python.md) for the interface.

## Windows convenience script

`compile.bat [Release|Debug|RelWithDebInfo|MinSizeRel]` builds a shared core in
`build/local/<configuration>` (Release by default), using the active Python
environment. Each configuration has its own `lib/` and `bin/`, so Debug and
Release outputs remain separate. The script enables SIMNRA, Python, examples and
benchmarks, and runs all tests including licensed execution. SIMNRA must be
installed. It pins CMake to the active `python` and checks NumPy, pybind11,
pywin32 and tqdm in isolated mode before compiling. Install these inside the
active conda environment, rather than its user site. Set the `ONNXRUNTIME_ROOT` environment variable to include ONNX
inference; otherwise the script builds with inference disabled. It does not
copy build artifacts into the source package.
