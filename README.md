# iBeamLab

iBeamLab combines a C++20 core with a Python interface for ion-beam analysis.
It supports simulation, reproducible training-data generation, model training
and packaging, and forward/inverse model inference.

- **Simulation:** a synthetic dummy backend and Windows SIMNRA COM integration.
- **Data generation:** multilayer samples, independent measurements, reproducible
  composition/thickness sampling, and streaming sharded datasets.
- **Training and export:** optional PyTorch utilities for custom models and
  standalone ONNX graphs packaged with their scientific metadata.
- **Inference:** parameter-to-spectra forward models and spectra-to-elemental-
  depth-profile inverse models, including multiple measurement spectra.
- **Processing:** calibrated rebinning, exposure scaling, and pileup operations.

Fitting is not implemented.

## Getting started

Run all commands from the repository root.

### Python

Install the package:

```console
python -m pip install .
```

For optional training dependencies, use `python -m pip install ".[training]"`.
The default build includes the dummy simulator, generation and package I/O.
Enable ONNX Runtime for model inference and Windows SIMNRA for real simulation;
see the [build guide](docs/building.md).

```python
import ibeamlab as ibl

experiment = ibl.Experiment(
    ibl.Sample([ibl.Layer(100, {"Si": 0.5, "O": 0.5})]),
    ibl.ExperimentalSetup([
        ibl.Measurement("RBS", ibl.Beam("H", 2000)),
    ]),
)
result = ibl.DummySimulator(channels=5000).simulate(experiment)
counts = result["RBS"].counts
```

Physical units are keV, seconds, and `1e15 atoms/cm2`. Measurements can use
different beam energies and detectors while sharing the same sample.

### C++

Configure, build and test the same root project:

```console
cmake -S . -B build/local -DCMAKE_BUILD_TYPE=Release
cmake --build build/local --config Release --parallel
ctest --test-dir build/local -C Release --output-on-failure
```

The default C++ build has no SIMNRA, ONNX, Python or download requirement.
Install with `cmake --install build/local --config Release --prefix <prefix>`.
C++ consumers use `find_package(ibeamlab CONFIG REQUIRED)` and link
`ibeamlab::ibeamlab`.

## Examples

[DataGeneration](examples/DataGeneration/README.md) contains a runnable
TOML-driven multilayer workflow. It supports predefined compounds, shared
random element sets, independent layerwise element sets, and pure samples.

```console
python examples/DataGeneration/generate_multilayer.py --validate-only
```

Use `--backend dummy` for synthetic data without SIMNRA. The example guide
provides small-budget smoke-run commands and explains the generated artifacts.

## Project structure

```text
CMakeLists.txt          Project options and build entry point
pyproject.toml          Python packaging, using the root CMake project
README.md               Project overview
cmake/                  Dependency discovery and installed C++ package configuration
cpp/
  CMakeLists.txt        Library, bindings, examples and test targets
  include/ibeamlab/     Public C++ API
  src/                 C++ implementation
  python/              Compiled Python bindings
  tests/               C++ and binding integration tests
ibeamlab/               Python package
tests/                  Python wrapper and workflow tests
examples/               Runnable workflows and their local instructions
docs/                   Project documentation
xnra/                   SIMNRA reference projects
old/                    Archived reference implementation
```

The root CMake file defines configuration once and delegates target definitions
to `cpp/CMakeLists.txt`. Python wheels use that same entry point.

## Documentation

- [Building and installation](docs/building.md)
- [Python interface](docs/python.md)
- [C++ API and scientific contracts](docs/cpp.md)
- [Testing and CI](docs/testing.md)
- [Dataset and model format](docs/model-format.md)
- [SIMNRA configuration and validation](docs/simnra.md)
- [Development checklist](docs/development.md)
