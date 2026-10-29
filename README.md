# iBeamLab: Ion-beam simulation, ML training, and deployment in Python and C++

<div align="center">
  <img src="logo/ibeamlab_logo.png" alt="iBeamLab logo" width="600" />
  <p><strong>From layered samples to simulation, training, and deployed inference.</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10 or newer" />
    <img src="https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white" alt="C++20" />
    <img src="https://img.shields.io/badge/Training-PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch training" />
    <img src="https://img.shields.io/badge/Inference-ONNX_Runtime-005CED?logo=onnx&logoColor=white" alt="ONNX Runtime inference" />
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="MIT license" /></a>
  </p>
</div>

iBeamLab provides a unified workflow from ion-beam simulation to machine-learning
inference in external applications. Python provides the main interface; a C++20
library handles simulation backends, dataset storage, spectrum processing, and
inference.

1. **🔬 Simulate and generate data.** Describe layered samples and experimental
   setups, then generate reproducible, sharded datasets using SIMNRA.
2. **🧠 Train a model.** Train a forward surrogate with the supplied example,
   or build your own forward or inverse architecture using the framework's
   datasets and scientific metadata.
3. **📦 Package and deploy.** Export an ONNX graph and its scientific contract
   into one model package, then use that same file in Python or C++ applications.

**Train once, deploy in either language.** A package carries the graph, units,
input/output schema, reference experimental setup, and preprocessing and
postprocessing pipelines. Applications supply physical inputs and receive labelled
spectra or elemental depth profiles without recreating the training architecture.
Exported inference requires neither PyTorch nor SIMNRA.

Multiple measurements can share a sample while using different beam energies and
detectors. Each forward output spectrum can have its own postprocessing pipeline;
each inverse input spectrum can have its own preprocessing pipeline. CPU and CUDA
inference share the same API.

Fitting is planned and is not yet implemented. The current training example is a
**forward surrogate**; inverse inference and package contracts are available for
custom models.

## 🧭 Start with your task

| Goal | Starting point |
| --- | --- |
| Define a sample and simulate spectra | Simulation example below; [Python guide](docs/python.md) |
| Generate multilayer training data | [DataGeneration guide](examples/DataGeneration/README.md), [configuration](examples/DataGeneration/multilayer-generation.toml), and [script](examples/DataGeneration/generate_multilayer.py) |
| Train a sample-to-spectra surrogate | [iBeamNet example](examples/ibeamnet/README.md) and [training notebook](examples/ibeamnet/train/train.ipynb) |
| Deploy a forward or inverse model | Deployment examples below; [Python inference guide](docs/python.md) |
| Define a custom model's scientific contract | [Dataset and model format](docs/model-format.md); [C++ contracts](docs/cpp.md) |
| Build and integrate the C++ library | [Build guide](docs/building.md) |
| Run tests or configure CI | [Testing guide](docs/testing.md) |

## 🛠️ Install and build

Use Python 3.10 or newer, CMake 3.24 or newer, and a C++20 compiler. On Windows,
install Visual Studio Build Tools with the C++ workload. Run commands from the
repository root in your chosen Python environment.

For an installed Python package:

```console
python -m pip install .
```

For training and plotting, use `python -m pip install ".[training,plot]"`.
Install `jupyterlab` or a compatible notebook environment for interactive training.

SIMNRA and ONNX inference are optional build features, disabled by default.
Installing Python `onnxruntime` alone does not enable inference in the C++
extension. See the [build guide](docs/building.md) for SDK configuration and
CPU/CUDA runtime dependencies.

For the Windows source-tree workflow with SIMNRA installed:

```powershell
python -m pip install numpy pybind11 pywin32 tqdm
.\compile.bat --release
python examples/DataGeneration/generate_multilayer.py --extension-dir build/local/Release/bin --validate-only
```

The script builds and tests the library and extension in
`build/local/Release`. It enables SIMNRA, and enables ONNX inference when
`ONNXRUNTIME_ROOT` points to an SDK. For a CUDA SDK, also set
`IBEAMLAB_ONNX_PROVIDER=CUDA`; see the build guide for details.

For interactive source-tree Python use, add the build's `bin` directory to
`PYTHONPATH`. Restart Python or notebook kernels after replacing the extension.

## 🔬 Define and simulate an experiment

Layers are ordered from the surface toward the substrate. Composition entries
are atomic fractions and must sum to one. Each measurement has its own beam and
detector settings.

```python
import ibeamlab as ibl

experiment = ibl.Experiment(
    ibl.Sample([
        ibl.Layer(100, {"Si": 0.5, "O": 0.5}),
    ]),
    ibl.ExperimentalSetup([
        ibl.Measurement("RBS", ibl.Beam("H", 2000)),
    ]),
)

result = ibl.DummySimulator(channels=5000).simulate(experiment)
counts = result["RBS"].counts
```

The dummy backend produces synthetic spectra for checking workflows. For physical
simulation on Windows, associate each measurement with a SIMNRA reference project:

```python
method = ibl.SimnraMethod("RBS", "reference.xnra", "H")
with ibl.SimnraSimulator([method], workers=4) as simulator:
    result = simulator.simulate(experiment)
```

Reference projects preserve measurement-specific geometry and cross-section
choices. The experiment and reference must describe compatible conditions.
See [SIMNRA configuration and validation](docs/simnra.md).

## 🗃️ Generate and read training data

The [multilayer example](examples/DataGeneration/README.md) supports predefined
compounds, one randomly selected element set shared across layers, and independently
selected element sets for each layer. Configure their fractions, element counts,
thickness ranges, samples per layer system, pure samples, seeds, and detector
references in its TOML file.

```powershell
python examples/DataGeneration/generate_multilayer.py --extension-dir build/local/Release/bin --validate-only
python examples/DataGeneration/generate_multilayer.py --extension-dir build/local/Release/bin
```

Use `--backend dummy` for synthetic generation. Progress is reported separately
for each layer system. Output includes configuration records and sharded datasets;
destinations must be new to prevent accidental overwrites.

For a small parameter study directly in Python:

```python
plan = ibl.GenerationPlan(experiment, [
    ibl.vary.layer_thickness(layer=0, bounds=(50, 150)),
])
rows = plan.sample_uniform(100, seed=42)
summary = plan.generate(rows, ibl.DummySimulator(), "datasets/example")

dataset = ibl.open_dataset(summary.path)
for batch in dataset.iter_batches(32):
    pass  # Process records in bounded batches.
```

Use streaming iteration for large collections. Convenience properties such as
`dataset.parameters` and `dataset.spectra("RBS")` materialize arrays in memory.

## 🧠 Train a forward surrogate

The [iBeamNet example](examples/ibeamnet/README.md) implements a recurrent forward
surrogate that predicts detector spectra from elemental areal densities. Each
layer contributes one value per element: thickness multiplied by atomic fraction.
There is no redundant total-thickness input column.

The network shares its layer recurrence across the sample and supports any
positive number of layers with a fixed experimental setup. Training layer counts
do not impose a deployment limit. Multiple detector heads predict their own
spectra, with independent scaling and postprocessing.

Start with [train/train.ipynb](examples/ibeamnet/train/train.ipynb), or edit
[surrogate.toml](examples/ibeamnet/surrogate.toml) and run:

```powershell
python examples/ibeamnet/run.py prepare --extension-dir build/local/Release/bin
python examples/ibeamnet/run.py train
python examples/ibeamnet/run.py export --extension-dir build/local/Release/bin
```

Preparation streams completed datasets into disk-backed arrays. Training uses an
explicit staged learning-rate schedule, keeps the best validation checkpoint,
and supports resuming optimizer and random states. The default schedule is
25 epochs at `5e-4`, 90 at `1e-4`, then 50 at `1e-5`.

Export creates one `forward.ibm` with dynamic batch and layer axes, and verifies
predictions against PyTorch. Native verification requires an ONNX-enabled build.
The architecture lives in the example; iBeamLab's package and inference APIs
support custom model types independently of it.

## 📦 Deploy in Python or C++

All current dataset and model packages use **format 1**. Model files use the
`.ibm` extension and contain a standalone ONNX graph plus scientific metadata.
The same package is consumed by both language interfaces.

For a forward package:

```python
forward = ibl.load_model("forward.ibm")
prediction = forward.predict(experiment)
counts = prediction["RBS"].counts

# Layer-input packages also accept physical elemental areal densities:
# prediction = forward.predict_areal_density(ad)
# ad: contiguous float32 [layers, elements], in packaged element order.
```

For an inverse package created by your model exporter:

```python
inverse = ibl.load_model("inverse.ibm")
measurement = ibl.InverseInput(
    setup=experiment.setup,
    spectra={"RBS": measured_counts},
)
profile = inverse.predict(measurement)
densities = profile.values  # [layer, element], in physical areal-density units.
```

Here `measured_counts` contains your measured spectrum. Inverse metadata declares
the maximum output layer count and ordered elements. Forward layer-input metadata
does not impose a maximum layer count. Both model types apply their declared
scientific processing pipelines.

CPU inference is the default. To select a CUDA-enabled build explicitly:

```python
options = ibl.InferenceOptions(execution_provider="cuda", device_id=0)
forward = ibl.load_model("forward.ibm", options=options)
```

C++ applications install the library and use its exported CMake target:

```cmake
find_package(ibeamlab CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE ibeamlab::ibeamlab)
```

The C++ inference API supports reusable caller-owned results and workspaces.
Use one model instance per worker. See the [C++ guide](docs/cpp.md) for package
loading, physical inputs, and allocation-conscious inference, and the
[build guide](docs/building.md) for runtime deployment.

## 📐 Units and ordering

| Quantity | Convention |
| --- | --- |
| Beam energy, energy spread, detector resolution | keV |
| Layer thickness and elemental areal density | `1e15 atoms/cm2` |
| Acquisition times | seconds |
| Composition and isotope fractions | Unitless fractions |
| Layers | Surface to depth |
| Parameter, element, and spectrum ordering | Declared dataset/package schema |

Areal thickness is an atomic inventory per unit area. Converting it to geometric
thickness requires a material number density.

## 🧩 Repository and documentation

| Path | Contents |
| --- | --- |
| `CMakeLists.txt`, `pyproject.toml` | Shared build entry point and Python packaging |
| `cpp/` | C++ library, public headers, compiled bindings, examples, and tests |
| `ibeamlab/` | Python framework API, metadata, datasets, and export utilities |
| `examples/DataGeneration/` | Configurable multilayer generation workflow |
| `examples/ibeamnet/` | Forward surrogate training, notebook, and deployment example |
| `tests/`, `.github/workflows/` | Python tests and automated build/integration checks |
| `docs/` | [Building](docs/building.md), [Python](docs/python.md), [C++](docs/cpp.md), [testing](docs/testing.md), and [file formats](docs/model-format.md) |
| `xnra/` | SIMNRA reference projects |
| `old/` | Archived reference implementation |

For a minimal C++ build and test run:

```console
cmake -S . -B build/minimal -DCMAKE_BUILD_TYPE=Release
cmake --build build/minimal --config Release --parallel
ctest --test-dir build/minimal -C Release --output-on-failure
```

The minimal build needs neither SIMNRA nor ONNX Runtime. Optional integration
tests are enabled with their corresponding build features; see the testing guide.

Released under the [MIT license](LICENSE).
