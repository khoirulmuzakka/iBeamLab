# iBeamLab C++ library

This is an independent C++20 project. `native/` and the existing Python package
are unchanged reference implementations. Fitting is intentionally not implemented.

See [TESTING.md](TESTING.md) for the implemented unit/integration suites and CI jobs.

## Build and test

```console
cmake -S cpp -B cpp/build
cmake --build cpp/build --config Release --parallel
ctest --test-dir cpp/build -C Release --output-on-failure
```

The default build needs no SIMNRA, ONNX, Python, or network downloads. Enable
`IBEAMLAB_ENABLE_SIMNRA=ON` on Windows to compile the licensed COM backend.
Enable `IBEAMLAB_ENABLE_ONNX=ON` and set `ONNXRUNTIME_ROOT` to a matching C/C++ SDK
for packaged inference. The build copies the matching Windows runtime beside
executables. Python's `onnxruntime` wheel is not the C++ SDK.

Other switches: `BUILD_SHARED_LIBS`, `IBEAMLAB_BUILD_EXAMPLES`,
`IBEAMLAB_BUILD_BENCHMARKS`, `IBEAMLAB_BUILD_PYTHON`, and
`IBEAMLAB_SANITIZERS` (GCC/Clang ASan+UBSan). Python bindings require pybind11
and NumPy. Set `Python_EXECUTABLE` to select an environment.
The ONNX-enabled Python test also requires PyTorch and ONNX for export parity.

Install with `cmake --install cpp/build --config Release --prefix <prefix>`.
Consumers use `find_package(ibeamlab CONFIG REQUIRED)` and link
`ibeamlab::ibeamlab`. Put the installed `bin/` on the runtime search path when
using a shared build or ONNX on Windows. Static ONNX builds install its import
library; runtime dependencies must still accompany the consuming application.
An independent installed-consumer test checks this integration.

## Modules

- `core`: shared samples, independently configured measurements, parameter targets,
  validation, units, owning spectra and non-owning views.
- `processing`: calibrated rebinning and pair-sum pileup application/removal.
- `simulation`: deterministic dummy spectra and persistent worker-owned SIMNRA COM sessions.
- `datasets`: checksummed sharded storage, transactional completion and streaming cursors.
- `generation`: physical parameter matrices, uniform sampling, failure policies and progress.
- `model`: shared package schema and embedded ONNX payload.
- `inference`: forward experiment-to-spectra and inverse spectra-to-EDP APIs.

All modules currently compile into one library. Public headers have no ONNX or
SIMNRA dependency. Model instances and dataset cursors are owned by one caller;
use separate instances per worker. The SIMNRA batch method uses its configured
worker pool and preserves input/measurement order. Target rebuilds append and populate
replacement layers before removing the original prefix, preserving cross-section
choices while elements remain present. Matching topology is normally reused;
`SimnraConfig.reuse_target_topology=false` forces rebuilding for diagnostics.
Pileup follows the reference project setting; `nopu` references are checked to
have it disabled. Cancellation is cooperative
between measurements; an active external COM calculation cannot be interrupted.

## Scientific and allocation contracts

Counts use contiguous `double` storage and represent integrated counts per
channel, not count density. ONNX tensors use float32. Calibration describes
channel centers `E(i)=offset+linear*i+quadratic*i*i`; edges are evaluated at half
integers. All edges must be finite and strictly increasing. Rebinning assumes
uniform density inside each source bin. It conserves counts over common support;
counts outside the destination support are discarded without renormalization.
`RequireFull` rejects a destination grid not covered by the source.

Rebinning into caller-owned output performs no allocations. Pileup application
uses a reusable FFT workspace and allocates only when its capacity grows.
The benchmark checks allocation counts after warmup at 5,000 channels. Inverse
pileup currently uses a quadratic recurrence: optimize it only after profiling.
Output buffers must not overlap input buffers. C++ views are valid only while
their owners remain alive and their buffers are not resized.
Python `*_into` processing functions require contiguous float64 NumPy arrays,
reject implicit conversion, and write into caller-owned output. Other Python
vector conversions are convenience operations and may copy.

Pileup is the documented pair-sum model in `processing/pileup.h`. It requires
positive acquisition times with live time <= real time and occupancy
`tau*total/real_time < 1`. Output includes full `2*N-1` support. Removal requires
that full support and verifies reconstructed counts. Truncated or noisy data
may have no supported inverse and are rejected. This model sums channel indices;
its applicability to an instrument must be validated separately.

Generation reuses a working experiment; it never normalizes concentrations
implicitly. Supply coupled composition rows explicitly. A complete dataset is
published only after all requested rows are accepted, recorded as failures, or
discarded. Cancellation and Stop failures leave an incomplete directory. A new
run rejects existing output directories. Caller-supplied rows are currently
resident; generated spectra and writer memory are bounded by one shard.

## Python and training/export

The independent distribution is `ibeamlab-cpp`, but its import name is
`ibeamlab`; use a separate environment from the legacy package.

```console
python -m pip install ./cpp
```

For inference, pass CMake definitions at install time (using a suitable SDK):

```console
python -m pip install ./cpp -Ccmake.define.IBEAMLAB_ENABLE_ONNX=ON -Ccmake.define.ONNXRUNTIME_ROOT=<sdk>
```

Install the training extras with `python -m pip install "./cpp[training]"`.
`export_package()` exports a custom PyTorch model with a flat float32
`[batch,width]` input and output, embeds its schema, and verifies graph parity
at two batch sizes when ONNX is enabled. Metadata preprocessing runs outside
the graph, so the export example must already be preprocessed.

`examples/train_and_export.py` demonstrates seeded generation, train/validation/
test splits, training-only normalization, best-checkpoint selection, package
export and physical inference parity. `examples/generate.cpp` demonstrates
reproducible C++ generation with the dummy backend. These examples validate
software behavior; the dummy backend is not a physical simulator.

## Model metadata and multiple measurements

`include/ibeamlab/model/metadata.h` defines model schemas and ordered processing
pipelines. `model/package.h` handles package reading/writing. Metadata stores
exactly one `ForwardModelMetadata` or `InverseModelMetadata`; direction follows
that active variant.

Both directions support multiple independent measurements. `SpectraSchema`
contains an ordered list of labels and channel counts. Each label resolves to
its own beam, calibration, resolution and acquisition settings. Spectra may have
different lengths and callers may provide them in any order: packing and output
ordering follow the schema. `width()` and `offset(label)` describe graph slices.
The current graph interface uses one flat float32 input/output tensor; multiple
spectra are concatenated into its channel dimension.

Forward metadata declares `input_parameters`, `output_spectra`, a tensor
`preprocessing` pipeline, and per-measurement `postprocessing`. Inverse metadata
declares `reference_setup`, `input_spectra`, EDP elements/layers/uncertainty,
per-measurement `input_preparation`, tensor `preprocessing`, and EDP `postprocessing`.
Shared metadata contains tensor names and provenance.

```python
specification = ibl.ForwardModelMetadata()
specification.reference = experiment  # Measurements labelled RBS and NRA
specification.input_parameters = plan.parameters
specification.output_spectra = ibl.SpectraSchema([
    ibl.SpectrumSpec("RBS", 5000),
    ibl.SpectrumSpec("NRA", 3000),
])
specification.preprocessing = [
    ibl.MinMaxScale(minimum=[100], maximum=[1000]),
]
specification.postprocessing = [
    ibl.SpectrumProcessing("RBS", [
        ibl.MultiplyCounts(1000), ibl.ExposureScale(), ibl.Rebin(),
        ibl.Pileup(tau_seconds=0.0000004),
    ]),
    ibl.SpectrumProcessing("NRA", [ibl.MultiplyCounts(1000)]),
]
metadata = ibl.ModelMetadata(specification)
```

This illustrative scaler broadcasts its coefficients; use one fitted coefficient
per ordered parameter when parameters have different ranges. Available tensor
steps are `Log1p`, `MinMaxScale`, `StandardScale`, `MultiplyScale` and
`CustomTransform`. Scalar coefficients broadcast; otherwise their width must
match the tensor. Zero min-max spans and nonpositive standard deviations are
rejected. Pipelines never fit their own coefficients or silently normalize inputs.

Spectrum steps are `MultiplyCounts`, `ExposureScale`, `Rebin`, `Pileup` and
`RemovePileup`. Their order is authoritative. Exposure scaling maps reference
to requested exposure for forward models and measured to reference exposure
for inverse models. Rebinning follows the corresponding destination calibration;
zero `channels` selects the graph schema length. Pileup operations use requested/
measured acquisition times by default; `AcquisitionSource.REFERENCE` selects the
reference times explicitly. Full pileup support extends N channels to 2*N-1.
EDP `ScaleEdp` steps scale means and standard deviations, leaving probabilities
unchanged. An empty pipeline means identity.

Custom tensor transforms store an identifier, version and named numeric
parameters. A consuming C++ application registers a thread-safe, fixed-width,
in-place implementation with `processing::register_custom_transform` from
`processing/transforms.h`. Unknown implementations fail when loading an inference
model. Python callbacks are not serialized into deployment packages.

## Deployment contract

Packages use format 1 and contain only the active schema and its processing
pipelines. See [formats.md](formats.md). The graph has fixed feature widths and
may use a dynamic batch axis. External-data ONNX graphs and variable layer-count
graph inputs are not supported. Inverse matrices use surface-to-depth, layer-major
ordering; optional output layout is `[means, probabilities, posterior std]`.
Mean/std are returned in `1e15 atoms/cm2`; probabilities are dimensionless.

Prepared inverse input skips `input_preparation` but still runs tensor
preprocessing and EDP postprocessing. Operations embedded in ONNX must not also
be declared as external pipeline steps. Changes in beam/resolution require
trained forward parameter targets; inverse inference rejects them. Calibration
and exposure differences require corresponding explicit processing steps.
High-level batches reuse a session and execute records sequentially.

For repeated forward predictions, use `predict(experiment, result)` with the same
caller-owned result to reuse its spectrum buffers. Parameter locations, permitted
changes and output processing links are resolved at model construction. Per-call
validation compares fields directly without copying or serializing experiments.
An empty output pipeline converts graph values directly into the result spectrum.
The same metadata supports one or several detectors without an extra spectrum
copy for the single-detector case. Returning a new result by value still requires
storage for that result. Enable benchmarks to check C++ allocations after warmup;
these checks do not measure all allocations inside ONNX Runtime.

## SIMNRA reference verification

The `xnra/` folder in the repository contains reference projects and stored
spectra. With SIMNRA and the iBeamLab Python extension enabled, CTest compares
all five directly; no separate expected dataset is needed:

```console
ctest --test-dir cpp/build -C Release -R simnra-reference --output-on-failure
```

This test uses NumPy and pywin32 for an independent COM baseline, loads isotope
mixtures correctly from the XNRA representation, checks saved spectra against an
untouched recalculation, and compares the simulator's two workers and session
reuse. Reference comparisons use rtol=1e-5 and atol=1e-5; worker repeatability
uses rtol=1e-10 and atol=1e-10. The originals are never written. Without registered
SIMNRA COM the test explicitly skips. See `SIMNRA_VALIDATION.md` for observed
failures; stored-curve auditing is optional because saved curves can be outdated. Fresh
backend comparisons, rebuilds and parameter-update checks remain mandatory.
Enable `IBEAMLAB_CHECK_SAVED_SIMNRA_SPECTRA=ON` for a strict stored-curve audit.

Linux sanitizer and clean-runner checks are configured in CI and require running
there; they have not been executed on this Windows host.
