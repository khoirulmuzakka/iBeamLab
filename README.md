# iBeamLab: Ion-beam simulation, ML training, and deployment in Python and C++

<div align="center">
  <img src="logo/ibeamlab_logo.png" alt="iBeamLab Logo" width="600" />
</div>


iBeamLab provides a unified workflow from ion-beam simulation to machine-learning
inference in external applications:

1. **Simulate and generate data.** Define layered samples and experimental
   settings, then generate reproducible training datasets using SIMNRA.
2. **Train a model.** Use the supplied PyTorch models to predict spectra from
   sample parameters or recover elemental depth profiles from spectra. Custom
   architectures can use the same datasets and model-package contract.
3. **Deploy in Python or C++.** Export the trained model as an iBeamLab package
   and load that same package in an external Python or C++ program through
   iBeamLab's inference module.

**Train once, deploy in either language.** The exported package carries the ONNX
model, preprocessing, units, input/output schema, and reference experimental
settings. External applications supply physical inputs and receive spectra or
elemental depth profiles through the inference API. They can load the package
without recreating the model architecture or preprocessing, and without the
training notebook or PyTorch.

The external program does not need to know the ML model's internal architecture,
layers, or weights: ONNX represents the computation graph, and ONNX Runtime
executes it. iBeamLab adds the scientific interface around that graph, including
input preparation, physical units, experimental corrections, and structured
outputs. Deployment therefore goes beyond importing model weights: the program
loads a complete inference package and interacts with its declared inputs and
outputs, while the model implementation remains inside the package.

The main interface is Python. A C++20 library handles simulation backends,
dataset storage, spectrum processing, and ONNX inference; it is also available
to native applications. SIMNRA simulation requires Windows and an installed
SIMNRA COM backend. Dataset access, dummy simulation, and packaged inference do
not require SIMNRA. PyTorch is needed for training, not for running exported models.

## Start with your task

| Goal | Starting point |
| --- | --- |
| Define a sample and simulate its spectra | Python example below; [quick start](docs/getting-started.md) |
| Generate a reproducible multilayer training collection | [generation configuration](examples/multilayer-generation.toml) and [generation script](examples/generate_multilayer.py) |
| Train a spectra-to-depth-profile inverse model | [train_ibanet.ipynb](examples/train_ibanet.ipynb); [IBAnet guide](docs/ibanet.md) |
| Train a sample-to-spectra forward surrogate | [train_lrn_multilayer.ipynb](examples/train_lrn_multilayer.ipynb); [LRN guide](docs/lrn.md) |
| Run a trained model in Python or C++ | Model-package examples below; [package format](docs/model-package.md) |
| Explore spatial scans and compare reconstructions | [simulated_NMC.ipynb](examples/simulated_NMC.ipynb) and [compare_spatial_scan_soc.ipynb](examples/compare_spatial_scan_soc.ipynb) |

## Install from this checkout

Use Python 3.10 or newer, CMake 3.24 or newer, and a C++20 compiler. On Windows,
use Visual Studio Build Tools with the C++ workload. Run these commands from the
repository root in your chosen Python environment:

```console
python -m pip install .
```

For training and plotting:

```console
python -m pip install ".[training,plot]"
```

To run the notebooks, also install a notebook environment such as JupyterLab
and select the same Python environment as the notebook kernel. Some analysis
notebooks additionally use the Python `onnxruntime` package:

```console
python -m pip install jupyterlab onnxruntime
```

The build downloads dependencies when they are not available locally. On
Windows x64, CMake can fetch the pinned ONNX Runtime distribution. On other
platforms, provide an ONNX Runtime installation through `ONNXRuntime_ROOT`;
without it, the build reports that packaged inference is unavailable. Python
`onnxruntime` is separate from the runtime used by the native library.

After rebuilding the native extension, restart Python processes and notebook
kernels before using the new API.

## Define and simulate an experiment

Start by describing the sample and the detectors used to measure it. Layers are ordered from the surface toward the substrate. Composition entries
are atomic fractions and must sum to one.

```python
import ibeamlab as ibl

sample = ibl.Sample([
    ibl.Layer(500, {"Si": 0.5, "O": 0.5}),
])
experiment = ibl.Experiment(sample, [
    ibl.Detector("RBS", ibl.Beam("H", energy=2974), resolution=20),
])

spectra = ibl.DummySimulator(channels=1024).simulate(experiment)
print(spectra["RBS"].counts.shape)
```

`DummySimulator` produces synthetic test spectra for checking code and dataset
workflows. Use SIMNRA for physical simulations. A `SimnraMethod` associates a
spectrum label with its reference `.xnra` file:

```python
method = ibl.SimnraMethod("RBS", "reference.xnra")
with ibl.SimnraSimulator([method], workers=4) as simulator:
    spectra = simulator.simulate(experiment)
```

Reference files and detector settings must describe your intended measurement
methods. See the [quick start](docs/getting-started.md) for parameter studies and
SIMNRA usage.

## Generate and read training data

Turn the experiment into a training dataset by sampling its physical parameters
and simulating the resulting spectra. A `GenerationStudy` defines the experiment and the physical parameters that may
vary. This small example uses the dummy backend and varies layer thickness:

```python
study = ibl.GenerationStudy(experiment, [
    ibl.vary.layer_thickness(layer=0, bounds=(100, 1000)),
])
summary = study.generate(
    "datasets/example", simulator=ibl.DummySimulator(channels=1024),
    samples=1000, seed=42,
)
dataset = ibl.open_dataset(summary.path)
parameters = dataset.parameters       # [sample, parameter], study parameter order
counts = dataset.spectra("RBS")        # [sample, channel]
```

For multilayer collections, edit the TOML configuration to set element sampling,
layer counts, thickness policy, sample counts, detector reference files, output
location, and seeds. Validate the configuration before starting generation:

```console
python examples/generate_multilayer.py examples/multilayer-generation.toml --validate-only
python examples/generate_multilayer.py examples/multilayer-generation.toml
```

Paths in the configuration are resolved relative to its location. A run records
the original and resolved configurations, reference-file checksums, sampling
audits, and a generation summary. Each layer-count dataset contains a
`dataset.toml` manifest and binary shards. Existing output directories are
rejected to prevent overwrites.

Use `open_dataset()` to read these datasets rather than parsing shards yourself.
Spectra are padded at the high-channel end to the recorded length for each
label. Dataset arrays materialize records in memory; plan memory use when
working with large collections. See the [dataset format](docs/dataset-format.md).

## Train forward and inverse ML models

The generated datasets support two complementary tasks: forward models predict
spectra from a sample, while inverse models recover a sample profile from spectra.
The notebooks below train these models and export packages for deployment.

### IBAnet: spectra to elemental depth profiles

[train_ibanet.ipynb](examples/train_ibanet.ipynb) loads multilayer datasets,
constructs elemental areal-density targets, splits the data, trains the model,
restores the best validation checkpoint, and exports a model package with a
native/PyTorch parity check. Set the dataset location, maximum layer count,
architecture, target scale, and training schedule in the notebook.

IBAnet uses detector-specific residual CNN encoders with GroupNorm, channel
compression, and a fused spectrum context. A unidirectional GRU decodes layers
from surface to depth using that context and learned layer-position embeddings.
Shared heads predict a presence probability and one conditional Gaussian per
layer/element cell. Context and decoder inputs use LayerNorm. Earlier targets
or sampled outputs are not fed into the decoder.

The prediction is a spike-and-slab distribution: absence has probability
`1 - P`; presence has a Gaussian with mean `Y` and standard deviation `R`.
Its posterior mean is `P * Y`, and its full posterior variance is
`P * R**2 + P * (1 - P) * Y**2`.

Raw counts enter the model; `log1p` and fitted standardization are embedded in
it. For noise-free simulated counts, the notebook can draw Poisson noise before
preprocessing, with fresh training draws and fixed validation/test draws.
Disable this augmentation for data that already contain counting noise.
Targets are thickness times atomic fraction, scaled by one global
`TARGET_SCALE`. Keep that scale identical when comparing validation NLLs;
negative Gaussian NLL values are possible.

See the [IBAnet guide](docs/ibanet.md) for shapes, normalization, likelihood,
checkpoint compatibility, and export settings.

### LRN: sample parameters to spectra

[train_lrn_multilayer.ipynb](examples/train_lrn_multilayer.ipynb) trains a
layerwise recurrent forward surrogate. `LRNModel` consumes parameters in the
study's declared order and predicts the configured detector spectra. Its export
stores the physical sample/setup and input transforms so deployed callers can
supply physical experiments. See the [LRN guide](docs/lrn.md).

## Deploy trained models in external Python and C++ applications

The models trained above are deployed through a shared model-package contract. Export your trained
`IBAnet` or `LRNModel` to a ZIP, distribute that ZIP with your application, and
load it through iBeamLab inference. The exact same file works in both languages;
the consuming program needs iBeamLab with ONNX inference enabled, but does not
need PyTorch, the training notebook, or SIMNRA to run the exported model.

For an external Python application:

```python
import ibeamlab as ibl

model = ibl.load_model("trained-model.zip")
# Forward package: model.predict(experiment)
# Inverse package: model.predict(measurement), where measurement is InverseInput
```

For an external C++ application loading an inverse package:

```cpp
#include <ibeamlab/inverse_model.h>
#include <vector>

ibeamlab::inference::InverseModel model("ibanet.zip");
std::vector<ibeamlab::inference::InverseInput> measurements;
// Populate measurements with spectra and their experimental settings.
const auto predictions = model.predict(measurements);
// Each prediction.edp contains values, presenceProbability, and posteriorStd.
```

Forward packages use `ibeamlab::inference::ForwardModel` from
`<ibeamlab/forward_model.h>`. See the native integration instructions below for
linking and runtime dependencies.

Custom ML architectures can use the same deployment path by exporting ONNX and
creating package metadata that meets the [model-package contract](docs/model-package.md).
The inference module reads that contract rather than depending on a particular
Python model class. Inputs must match the packaged scientific schema and the
supported preprocessing and correction rules.

### Inverse inference: spectra to elemental depth profiles

An iBeamLab model ZIP contains ONNX weights and a TOML manifest describing the
input/output layout, physical reference setup, units, and transforms. Load an
inverse package and supply measured spectra together with their detector settings:

```python
model = ibl.InverseModel("ibanet.zip", threads=4)
measurement = ibl.InverseInput(
    spectra={"RBS": ibl.Spectrum("RBS", measured_counts)},
    detectors=measured_detectors,
)
result = model.predict(measurement)

mean = result.values
probability = result.presence_probability
std = result.posterior_std
print(result.elements, result.unit, mean.shape)
```

Here `measured_counts` is your raw count array and `measured_detectors` contains
your acquisition settings. Supply every spectrum label required by the package.
`predict()` performs the declared pileup correction, calibration rebinning, and
exposure scaling before applying embedded preprocessing. It requires coverage
of the training energy grid; changes in beam conditions or detector resolution
are not corrected. For already pileup-corrected spectra, set
`pileup_already_removed=True` on `InverseInput`.

For simulated/reference spectra already matched to the training setup, use:

```python
result = model.predict_prepared({"RBS": reference_spectrum})
```

`predict_prepared()` skips physical corrections but still applies the packaged
preprocessing. Supply counts without manually applying the model's log transform
or standardization. Both inference methods also accept batches.

All three output matrices have shape `[layer, element]`, ordered surface to depth
and according to `result.elements`. Mean and standard deviation use
`1e15 atoms/cm2`; probability is dimensionless. Sum mean densities across elements
to obtain each layer's areal thickness. Divide by that sum to obtain atomic
fractions for nonzero layers. Presence probability describes elemental presence,
not confidence in the accuracy of the atomic fraction.

New IBAnet exports use inverse package format **4** and provide actual
probabilities and full posterior standard deviations. Legacy format **3** inverse
packages remain readable: probability defaults to 1 for positive densities and
0 for exact zeros, std defaults to 0, and `result.uncertainty_predicted` is false.
Re-export a trained checkpoint to obtain predicted uncertainty; retraining is
not required solely for this export change. Deterministic model authors can
emit probability 1 and std 0 and declare that uncertainty is not predicted.

Small positive means may remain for absent elements and padded layers; export
does not impose a hard presence threshold. Keep the PyTorch checkpoint if you
need posterior sampling or conditional Gaussian parameters.

### Forward inference: sample parameters to spectra

```python
forward = ibl.ForwardModel("lrn.zip", threads=4)
predicted_spectra = forward.predict(experiment)
print(predicted_spectra["RBS"].counts)
```

Forward inference accepts physical experiments and applies the package's
parameter transforms and declared spectrum corrections. Forward packages use
format **2**. `ibl.load_model(path)` inspects a package and returns the appropriate
forward or inverse model. See [forward corrections](docs/forward-corrections.md)
for calibration, exposure, and pileup behavior.

## Units and ordering

| Quantity | Convention |
| --- | --- |
| Beam energy, spread, detector resolution | keV |
| Layer thickness and elemental areal density | `1e15 atoms/cm2` |
| Acquisition times | seconds |
| Composition and isotope fractions | Unitless fractions |
| Layers | Surface to depth |
| Parameter and element columns | Declared study/package order |

Areal thickness is an atomic inventory per unit area, not a geometric thickness
in nanometers. Converting it to a length requires a material number density.

## Build, test, and integrate the C++ library

Build the C++ library, Python extension, and tests from the repository root:

```console
cmake -S . -B build -DIBEAMLAB_BUILD_PYTHON=ON -DIBEAMLAB_BUILD_TESTS=ON
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
python -m pip install ".[dev,training,plot]"
python -m pytest
```

Install the C++ library with `cmake --install build --config Release --prefix <path>`.
A consumer uses the exported target and runtime-copy helper:

```cmake
find_package(ibeamlab CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE ibeamlab::ibeamlab)
ibeamlab_copy_runtime_dependencies(my_application)
```

The public API lives under `native/include/ibeamlab`. Python users can explicitly
import `ibeamlab.native` for low-level bindings. Native consumers must rebuild
when public structure layouts change. See [native architecture](docs/native.md)
for runtime dependencies, SIMNRA workers, build options, and C++ examples.

## Repository layout

| Path | Contents |
| --- | --- |
| `ibeamlab/` | Python API, configuration/sampling, and optional PyTorch models |
| `native/` | C++ library, Python bindings, tests, and native examples |
| `examples/` | Generation scripts, configuration, training and analysis notebooks |
| `docs/` | Workflow guides and dataset/model-package contracts |
| `tests/` | Python API and model tests |
| `cmake/` | Build and installed-consumer integration |

See [LICENSE](LICENSE) for the repository license.
