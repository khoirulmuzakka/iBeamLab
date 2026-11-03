# Building and installation

All builds start at the repository root. `CMakeLists.txt` defines project options
once and delegates library, extension and test targets to `cpp/CMakeLists.txt`.
Python packaging uses the same root project through `pyproject.toml`.
Use a fresh build directory for the unified layout, such as `build/cpu/Release`.

## C++ library

Requirements: CMake 3.24+ and a C++20 compiler. The default build is static,
enables C++ tests and native fitting through Minion. CMake locates Minion 2.0.1+
with `find_package(Minion CONFIG REQUIRED)`. Set `CMAKE_PREFIX_PATH` or
`Minion_DIR` to its installation; Eigen must also be discoverable as required by
that package. The ONNX Runtime C/C++ SDK is also required. An external
simulator and Python are needed only for their respective components.

```console
cmake -S . -B build/cpu/Release -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu/Release --config Release --parallel
ctest --test-dir build/cpu/Release -C Release --output-on-failure
cmake --install build/cpu/Release --config Release --prefix /path/to/install
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
| `IBEAMLAB_ONNX_PROVIDER` | CPU | SDK capabilities: CPU or CUDA; CUDA builds can run either provider |
| `IBEAMLAB_SANITIZERS` | OFF | GCC/Clang address and undefined-behavior sanitizers |
| `IBEAMLAB_REQUIRE_SIMNRA_TEST_RUNTIME` | OFF | Fail licensed tests if SIMNRA is unavailable |
| `IBEAMLAB_CHECK_SAVED_SIMNRA_SPECTRA` | OFF | Also require stored XNRA curves to match fresh calculations |

SIMNRA requires Windows and a local SIMNRA installation. Its worker COM
apartments default to MTA (`multithreaded_apartment=true`). See [simnra.md](simnra.md).

Every build requires native ONNX inference and a matching ONNX Runtime C/C++ SDK.
Configuration fails if either required dependency is unavailable. Python's
`onnxruntime` package is not a substitute for that SDK. Set `ONNXRUNTIME_ROOT`:

```console
cmake -S . -B build/onnx -DONNXRUNTIME_ROOT=/path/to/sdk
cmake --build build/onnx --config Release --parallel
```

Windows builds copy the matching `onnxruntime.dll` beside build executables.
Redistribute runtime dependencies alongside consuming applications. The
installed-consumer test checks C++ package discovery and linking.

For CUDA, provide a GPU ONNX Runtime SDK (with CUDA and shared provider
libraries) and configure a separate build:

```console
cmake -S . -B build/onnx-cuda -DIBEAMLAB_ONNX_PROVIDER=CUDA -DONNXRUNTIME_ROOT=/path/to/gpu-sdk -DIBEAMLAB_BUILD_PYTHON=ON
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

`compile.bat --release` selects CPU; add `--gpu` to select CUDA.
`--onnx-root` or `ONNXRUNTIME_ROOT` supplies an existing SDK.

## Python installation

Requirements: Python 3.10+, NumPy, CMake and a C++20 compiler. Pip installs the
scikit-build-core/pybind11 build dependencies in its isolated build environment.

```console
python -m pip install .
python -m pip install ".[training]"
```

The `training` extra adds PyTorch, ONNX and Python ONNX Runtime. The `examples` extra adds TOML
support on Python 3.10; Python 3.11+ includes it. Default wheels enable Python
bindings, a shared core and native ONNX inference. Tests/examples and SIMNRA
are disabled by default. Both native dependencies must be discoverable by CMake.

Provide the required SDK in a wheel build:

```console
python -m pip install . --config-settings "cmake.define.ONNXRUNTIME_ROOT=/path/to/sdk"
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

## Windows convenience scripts

Activate the Python environment containing NumPy, pybind11, pywin32, tqdm,
PyTorch, ONNX and Python ONNX Runtime, then run:

```powershell
.\compile.bat --release
.\compile.bat --release --gpu
```

The default is CPU. `--debug`, `--relwithdebinfo` and `--minsizerel` select other
configurations; `--parallel N` controls build parallelism. The script enables
SIMNRA, Python bindings, tests, examples and benchmarks. It runs the tests before
installing. SIMNRA must be installed for its licensed tests.

```text
.dependencies/
    onnxruntime/<version>/cpu/
    onnxruntime/<version>/cuda/
    minion/<version>/<configuration>/
build/
    cpu/Release/
    cuda/Release/
dist/
    cpu/Release/
    cuda/Release/
    gpu/ibeamlab-gpu-win-x64.zip
```

Dependencies are separate from disposable build directories. The build tool
uses `--onnx-root` or `ONNXRUNTIME_ROOT` first, then the selected build's cached
SDK path, then the pinned dependency cache (ONNX Runtime 1.25.1 for both providers). It downloads only when that cache
is absent. Explicit invalid paths fail rather than triggering a download.
`Minion_DIR` or `CMAKE_PREFIX_PATH` selects existing Minion/Eigen packages;
otherwise pinned versions are prepared in `.dependencies`. CMake performs no
downloads and requires both packages through `find_package`.

The Python extension and DLLs are in `build/cpu/Release/bin` (or the CUDA
counterpart). Add this directory to `PYTHONPATH` for source-tree usage. Restart
Python after replacing a loaded extension. Installed distributions include
headers, CMake package files, Python bindings and runtime DLLs.

### Optional GPU bundle

After a CUDA build, prepare a self-contained runtime bundle:

```powershell
.\package_gpu_runtime.bat --cuda-dir C:/CUDA/bin --cuda-dir C:/cuDNN/bin
```

Without `--cuda-dir`, the tool looks in the active environment's PyTorch/NVIDIA
package DLL directories and `CUDA_PATH`/`CUDNN_PATH`. Supply explicit directories
to avoid mixing incompatible installations. The tool includes the CUDA iBeamLab
build, GPU ONNX Runtime, Minion, Python extension and CUDA/cuDNN DLLs. The NVIDIA
driver remains a system prerequisite. SDK headers are not needed at deployment.

Packaging runs native CPU and CUDA inference probes with developer DLL search
paths removed. A failed probe prevents publication of the ZIP. The manifest
records the public-header/compiler ABI fingerprint, build configuration,
architecture, ONNX API/runtime version and per-file SHA-256 checksums. When a
CPU installation exists, its ABI/configuration/ONNX API must match. An ABI
fingerprint is a compatibility check, not proof of numerical equivalence.

Use one ONNX Runtime distribution per process. GPU runtime supports both CPU
and CUDA sessions, selected through `InferenceOptions`; do not mix CPU and GPU
copies of `onnxruntime.dll`. Model `.ibm` packages remain device-independent.

## Preparing the Minion dependency

Use an existing Minion installation, or prepare ordinary upstream installations
with the optional development/CI helper:

```console
python tools/build_minion.py --prefix .dependencies/minion/2.0.1/Release
cmake -S . -B build/cpu/Release -DCMAKE_PREFIX_PATH=/absolute/path/to/.dependencies/minion/2.0.1/Release
```

The helper installs pinned Minion and Eigen packages; it is separate from the
normal iBeamLab configure step. Existing source checkouts can be supplied with
`--minion-source` and `--eigen-source` for offline preparation. The normal build
never compiles a hand-selected subset of Minion sources.

Minion's shared runtime is copied beside Windows build executables and included
in Python installations. Static iBeamLab consumers discover Minion through the
installed iBeamLab CMake configuration; make its dependency prefix discoverable
when configuring the consuming application. Shared iBeamLab consumers need the
runtime library but do not inherit Minion headers or its CMake target.
