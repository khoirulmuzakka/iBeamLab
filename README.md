# iBeamLab: A unified framework for ion-beam simulation and analysis, ML model training, and deployment in Python and C++

<div align="center">
  <img src="logo/ibeamlab_logo.png" alt="iBeamLab logo" width="600" />
  <p><strong>Simulate spectra. Estimate elemental depth profiles. Develop and deploy ML models.</strong></p>
  <p>
    <img src="https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white" alt="C++20" />
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10 or newer" />
    <img src="https://img.shields.io/badge/Training-PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch training examples" />
    <img src="https://img.shields.io/badge/Inference-ONNX_Runtime-005CED?logo=onnx&logoColor=white" alt="ONNX Runtime inference" />
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="MIT license" /></a>
  </p>
</div>

iBeamLab brings **simulation, elemental depth profile estimation, and machine
learning** into one framework for ion-beam analysis. It provides a **C++20 library
and a Python API**, with shared physical descriptions, datasets, and model packages.
Use C++ to integrate analysis into an application, Python to explore experiments
and develop models, or both within the same workflow.

| Capability | What you can do |
| --- | --- |
| **Simulation** | Describe layered samples and experimental setups, and calculate spectra through an external simulator backend. SIMNRA is currently supported on Windows; the backend interface allows additional simulators. |
| **Fitting and EDP estimation** | Fit measured spectra to estimate elemental depth profiles and experimental parameters using physical simulators or forward ML surrogates. |
| **Machine learning** | Generate reproducible datasets, develop and train forward or inverse models, and package them for inference in external C++ and Python applications. |

## Shared data, C++ and Python APIs

C++ and Python use the same descriptions of samples, isotopes, experiments,
parameters, and spectra. Multiple measurements can share a sample while using
different beam energies, detector geometries, and acquisition settings.

Both APIs support simulation, fitting, dataset generation, spectrum processing,
and model inference. The training examples use PyTorch.

The same `.ibm` model package works in C++ and Python. It includes the ONNX model,
physical input/output definitions, and processing settings. External applications
can run inference without the training code, PyTorch, or SIMNRA, using an
ONNX-enabled iBeamLab build with compatible runtime dependencies.

## Install and build

Build from the repository root with CMake 3.24 or newer and a C++20 compiler.
Native fitting uses the installed Minion package by default; set
`CMAKE_PREFIX_PATH` to its installation. See the [dependency setup](docs/building.md#preparing-the-minion-dependency).
Python usage requires Python 3.10 or newer. On Windows, install Visual Studio
Build Tools with the C++ workload.

### C++ library

A minimal build and test run needs neither SIMNRA nor ONNX Runtime:

```console
cmake -S . -B build/minimal -DCMAKE_BUILD_TYPE=Release
cmake --build build/minimal --config Release --parallel
ctest --test-dir build/minimal -C Release --output-on-failure
cmake --install build/minimal --config Release --prefix install
```

Point your application's `CMAKE_PREFIX_PATH` to the installation prefix, then link
against the exported target:

```cmake
find_package(ibeamlab CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE ibeamlab::ibeamlab)
```

Public headers are under [cpp/include/ibeamlab](cpp/include/ibeamlab). See the
[build guide](docs/building.md) for optional backends and dependencies.

### Python package

```console
python -m pip install .
```

For training and plotting:

```console
python -m pip install ".[training,plot]"
```

Install JupyterLab or another notebook environment for interactive training.

ONNX Runtime inference and Minion fitting are required in every build.
SIMNRA remains an optional Windows backend.
Installing the Python `onnxruntime` package alone does not enable ONNX inference
in the compiled library. Both language interfaces use the configured C++ runtime.

### Windows source-tree workflow

With SIMNRA installed:

```powershell
python -m pip install numpy pybind11 pywin32 tqdm
.\compile.bat --release
python examples/DataGeneration/generate_multilayer.py --extension-dir build/cpu/Release/bin --validate-only
```

The script prepares cached dependencies, builds and tests the CPU library and
Python extension in `build/cpu/Release`, and installs to `dist/cpu/Release`.
Use `compile.bat --release --gpu` for a separate CUDA build. Native ONNX
inference is always included; SIMNRA is enabled by this Windows script.
See the [build guide](docs/building.md) for runtime dependencies. For interactive source-tree Python usage, add the build's `bin`
directory to `PYTHONPATH`; restart the kernel after replacing the extension.

## Simulation

Describe a sample as layers ordered from the surface toward the substrate.
Species use atomic fractions and either natural isotope compositions or explicit
isotope mixtures. Each measurement has its own beam, detector geometry,
calibration, resolution, and acquisition settings.

A small Python experiment using the synthetic backend:

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

`DummySimulator` checks workflows; it is not a physical simulator. For physical
simulation with SIMNRA, associate measurements with reference `.xnra` projects.
Reference projects retain backend-specific settings, including cross-section
choices; their geometry must agree with the experiment.

C++ backends implement
[`simulation::Simulator`](cpp/include/ibeamlab/simulation/simulator.h), allowing
additional simulators to participate in generation and fitting workflows.
See the [SIMNRA guide](docs/simnra.md) and [C++ guide](docs/cpp.md).

## Fitting and elemental depth profiles

Use [smooth profiles](docs/profiles.md) or explicit layers, and combine
measurement-specific backends through the [analysis API](docs/analysis.md).

Estimate a layered sample's elemental depth profile by comparing measured spectra
with predictions from a simulator or forward surrogate. Declare measured data,
fit parameters, physical bounds, composition constraints, and selected channel
regions through `MeasuredData` and `FitProblem`, then run `fit`.

Text/CSV readers support explicit file layouts, missing-channel masks, and
optional variances. Native ARRDE evaluates populations in batches and retains
buffers as population sizes change. Multiple measurements can constrain a shared
sample while retaining their individual experimental settings.

Fitting is available through the C++ library and Python API. See the
[fitting guide](docs/fitting.md), [Python example](examples/Fitting/fit_spectrum.py),
and [C++ fitting interfaces](cpp/include/ibeamlab/fitting).

## Machine learning: generate, develop, package, deploy

### Generate data

Parameter studies connect physical sample/setup parameters to a simulation
backend and write reproducible, sharded datasets. Store parameter definitions,
reference experiments, provenance, and simulation outcomes alongside the spectra.
C++ readers and Python streaming APIs support bounded-memory access.

The [multilayer generation example](examples/DataGeneration/README.md) supports
predefined element subsets, random element sets shared across layers, and
independent element sets per layer. Configure sampling budgets, layer counts,
composition fractions, thickness ranges, seeds, and detector references:

```powershell
python examples/DataGeneration/generate_multilayer.py --extension-dir build/cpu/Release/bin
```

Use `--validate-only` to check configuration or `--backend dummy` for synthetic
workflow checks. Output directories must be new. The
[C++ generation example](cpp/examples/generate.cpp) demonstrates the library API.

### Develop and train models

Use the datasets and scientific metadata to develop your own forward
(sample-to-spectra) or inverse (spectra-to-profile) model.

The supplied [iBeamNet example](examples/ibeamnet/README.md) demonstrates a
recurrent forward surrogate trained on elemental areal densities. Its shared
recurrence supports variable layer counts for a fixed experiment, with independent
detector outputs. Start with the [training notebook](examples/ibeamnet/train/train.ipynb).
It includes a staged learning-rate schedule, validation history, best-model
checkpointing, and resumable training.

This is one model-development example. The framework's metadata, export, and
inference contracts support custom architectures. Inverse inference and packaging
are available; the supplied training example is currently forward-only.

### Package and deploy

A format-1 `.ibm` model package contains a standalone ONNX graph and its scientific
metadata: physical schemas, units, element and detector ordering, isotope
assumptions, reference setup, numerical transforms, and physical processing.
Numerical training transforms are distinguished from inference preparation such
as rebinning, exposure scaling, and pileup. Detector pipelines are independent.

Load a forward package in Python:

```python
forward = ibl.load_model("forward.ibm")
prediction = forward.predict(experiment)
counts = prediction["RBS"].counts
```

Load the same package in C++:

```cpp
#include <ibeamlab/inference/models.h>

ibeamlab::inference::ForwardModel forward(
    ibeamlab::model::Package::open("forward.ibm"));
ibeamlab::simulation::Result prediction;
forward.predict(experiment, prediction);  // experiment: core::Experiment
```

Both interfaces provide forward and inverse inference with CPU or CUDA execution,
according to the configured build and runtime. Forward layer-input contracts do
not impose a maximum layer count; inverse metadata declares the maximum output
layer count. Model accuracy outside its training distribution must be evaluated
for the application.

C++ inference supports caller-owned output buffers and reusable workspaces.
Batched prediction executes a population in one graph call. Use a separate model
instance per concurrent worker. See the [C++ guide](docs/cpp.md) and
[Python guide](docs/python.md) for batch APIs, inverse inference, provider settings,
and allocation behavior; see the [model format](docs/model-format.md) for the
package contract.

## Units and ordering

| Quantity | Convention |
| --- | --- |
| Beam energy, energy spread, detector resolution | keV |
| Layer thickness and elemental areal density | `1e15 atoms/cm2` |
| Detector angles | degrees |
| Acquisition times | seconds |
| Composition and isotope fractions | Unitless fractions |
| Layers | Surface to depth |
| Parameter, element, and spectrum ordering | Declared dataset/package schema |

Areal thickness is an atomic inventory per unit area. Converting it to geometric
thickness requires a material number density.

## Repository and documentation

| Path | Contents |
| --- | --- |
| `CMakeLists.txt`, `pyproject.toml` | Build entry point and Python packaging |
| `cpp/` | C++ library, public headers, compiled bindings, examples, and tests |
| `ibeamlab/` | Python API, metadata, datasets, and export utilities |
| `examples/DataGeneration/` | Multilayer dataset generation |
| `examples/Fitting/` | Measured-spectrum fitting example |
| `examples/ibeamnet/` | Forward surrogate development, training, and deployment example |
| `docs/` | [Building](docs/building.md), [Python](docs/python.md), [C++](docs/cpp.md), [fitting](docs/fitting.md), [testing](docs/testing.md), and [formats](docs/model-format.md) |
| `tests/`, `.github/workflows/` | Python tests and automated build/integration checks |
| `xnra/` | SIMNRA reference projects |
| `old/` | Archived reference implementation |

Released under the [MIT license](LICENSE).
