# iBeamLab

iBeamLab is an open-source toolkit for generating ion-beam-analysis datasets,
preprocessing spectra, training machine-learning models in Python, and deploying
exported ONNX models from C++.

The native implementation is in `native/`. It is independent of Python and GUI
frameworks. Python bindings expose selected native APIs through
`ibeamlab._ibeamlab_cpp`.

## Native build

Requirements are CMake 3.24+, a C++20 compiler, and (on Windows) the Visual C++
runtime. SIMNRA support is Windows-only. CMake fetches pinned toml++, miniz, and
ONNX Runtime dependencies when they are not installed.

```bat
compile.bat Release
```

This places the C++ import library at `lib/ibeamlab.lib`. The Python extension
and its runtime DLLs are placed directly in the `ibeamlab/` package beside
`__init__.py`, with no `Release/` directory. Use `compile.bat Debug` for a debug
build.

Or configure directly:

```sh
cmake -S . -B build -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
cmake --install build --config Release --prefix install
```

Native C++ consumers use one installed package and one library:

```cmake
find_package(ibeamlab CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE ibeamlab)
```

Public headers use flat include paths such as `<ibeamlab/simulator.h>` and
`<ibeamlab/inference.h>`.

Options include `IBEAMLAB_BUILD_TESTS`, `IBEAMLAB_BUILD_PYTHON`,
`IBEAMLAB_ENABLE_SIMNRA`, `IBEAMLAB_ENABLE_ONNX`, `IBEAMLAB_FETCH_ONNX`,
`IBEAMLAB_BUILD_EXAMPLES`, `IBEAMLAB_BUILD_BENCHMARKS`, and
`IBEAMLAB_ENABLE_SANITIZERS`. Set `IBEAMLAB_SIMNRA_REFERENCE` to a usable `.xnra`
file before configuring to enable the optional installed-SIMNRA integration test.
Python bindings are disabled by default for C++ consumers. `compile.bat` enables
them explicitly for the local Python training environment.

`ibeamlab-generate-dummy OUTPUT [SAMPLES]` produces a deterministic headless test
dataset without Python or SIMNRA.

See [native architecture](docs/native.md), [dataset format](docs/dataset-format.md),
and [model packages](docs/model-package.md).
