# Native architecture

The public C++ API uses the `ibeamlab` namespace and C++20. Physical units are:
beam energy, spread and detector resolution in keV; layer thickness in
1e15 atoms/cm2; acquisition times in seconds; isotope mass in atomic mass units.

The single `ibeamlab` library contains these concrete responsibilities:

- sample structures and experimental setup;
- simulation requests/results and SIMNRA execution;
- parameter materialization and batch generation;
- streaming, sharded dataset storage;
- rebinning, pileup, and reversible transforms;
- safe TOML/ZIP model-package loading;
- long-lived ONNX Runtime inference sessions.

Consumers link only `ibeamlab`; the responsibilities above are C++ namespaces,
not separately deployed libraries.

After installing iBeamLab, a CMake consumer needs only the exported namespaced
target:

```cmake
find_package(ibeamlab CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE ibeamlab::ibeamlab)
ibeamlab_copy_runtime_dependencies(my_application)
```

The corresponding forward-inference entry point is intentionally small:

```cpp
#include <ibeamlab/forward_model.h>

ibeamlab::inference::ForwardModel model("trained-model.zip");
const auto& metadata = model.metadata();
ibeamlab::simulator::SimulationInput input{
    metadata.forward.sampleTemplate,
    metadata.forward.setupTemplate,
};
const auto result = model.predict({input});
```

The installed runtime bundle includes the iBeamLab shared library and, when
iBeamLab fetched ONNX Runtime itself, the matching ONNX Runtime shared library.
On Windows, `ibeamlab_copy_runtime_dependencies()` places these DLLs beside the
consumer executable. This also prevents an unrelated system-wide ONNX Runtime
DLL from shadowing the version used to build iBeamLab.

The shared-library ABI is versioned with the iBeamLab major version. Because
the value-oriented API deliberately exchanges C++ standard-library containers,
Windows consumers must use a binary-compatible MSVC toolset and C++ runtime.
The public headers use explicit visibility annotations; implementation symbols
are hidden instead of relying on automatic DLL symbol export.

SIMNRA workers are persistent. Each worker constructs, uses, and destroys its COM
objects on the same worker thread. Cancellations are cooperative. Batch results
retain request order and may contain structured per-item failures.

Public containers preserve insertion order. Concentrations and isotope fractions
must sum to one within an absolute tolerance of 1e-6; they are never silently
normalized. A missing isotope list means natural abundance. Spectrum counts use
32-bit floats for storage and ML interoperability; physical parameters use doubles.
Sample models and experimental setups round-trip through versioned TOML using
`sample::toToml`, `sampleModelFromToml`, and `experimentalSetupFromToml`.

`compile.bat Release` (or `Debug`) places a flat runtime bundle in `lib`; no
configuration-named subdirectory is created there.

ONNX inference defaults to one intra-op and one inter-op CPU thread. Graph
optimizations are opt-in through `InferenceOptions::enableGraphOptimizations`;
this keeps the default deterministic and avoids oversubscribing generation
workers. Applications may tune these values after benchmarking their model.

## Inverse EDP inference

Inverse packages use format version 3. Existing version 2 forward packages remain
supported; older inverse parameter packages must be re-exported. `outputEdp`
declares `maxLayers`, ordered `elements`, and the unit `1e15 atoms/cm2`.
Rows represent variable-thickness layers from surface to depth. ONNX tensors
remain rank two: `[batch, sum(inputSpectra.length)]` and
`[batch, maxLayers * elements.size()]`, with layer-major output flattening.
The package stores layer order, semantics, flattening, and trailing-zero padding
explicitly and rejects unsupported layouts. All template layers must contain
exactly the declared elements; their species order may differ from column order.

```cpp
#include <ibeamlab/inverse_model.h>

ibeamlab::inference::InverseModel model("inverse-edp.zip");
ibeamlab::inference::InverseInput input{
    {{"RBS_front", rbsCounts}, {"PIXE", pixeCounts}},
    measuredSetup,
};
// Set input.pileupAlreadyRemoved = true for spectra already pileup-free
// with live-time scaling undone (including simulations without pileup).
auto results = model.predict({input});
const auto& edp = results[0].edp;
float arealDensity = edp.values[0][0]; // layer 0, edp.elements[0]
auto sample = edp.toSample(model.metadata().inverse.sampleTemplate);
```

`predict()` accepts experimental spectra and their setup. It matches labels,
removes pileup when required, rebins the experimental calibration onto the
training grid, scales by reference/measured ParticlesSr, and applies packaged
input transforms before ONNX. The inverse output transform restores physical
areal densities before constructing the EDP. Input order is irrelevant.

Set `pileupAlreadyRemoved=true` only when all spectra in that input are already
pileup-free with live/real-time scaling undone; simulated spectra generated
without pileup also use this flag. The flag skips subtraction, not calibration,
exposure, or training transforms. Otherwise, packages declaring
`need_pileup_subtraction=true` use the measured real/live times and packaged
`pileup_fudge_factor_seconds` (default `0.4e-6` seconds). Missing subtraction flags
in older packages default to false. IBAnet exports default to true.

Both measured and reference ParticlesSr must be positive for setup-aware
prediction. Calibration edges must be finite and increasing; the measured
energy range must cover the complete reference range. Rebinning changes the
channel count to the packaged input length. Beam particle, energy, spread, and
resolution differences warn once, following the forward-model behavior; no
physical correction is available for them. Such mismatches can reduce accuracy.

Pileup removal requires the complete pileup tail. Cropped or inconsistent tails
are rejected. Already prepared training-reference spectra use the explicit
`predictPrepared(batch)` entry point (`predict_prepared()` in Python). It checks
packaged labels and lengths and skips physical corrections. Packaged transforms
still execute exactly once. IBAnet embeds log1p and normalization inside ONNX,
so callers must not repeat them.

Python applications can construct `InverseInput(spectra, detectors,
pileup_already_removed=False)` and call `InverseModel.predict(input)` or supply
a sequence for a batch. The previous spectra-only call is now
`InverseModel.predict_prepared(spectra)`.

Predicted densities must be finite and nonnegative; zero rows may only be trailing
padding. Invalid input or output throws and aborts the batch. An empty batch
returns an empty result. An entirely zero EDP is representable but cannot be
converted to a valid nonempty sample.

`toSample()` sums each row for thickness and divides elemental densities by the
row sum for concentrations. It removes trailing zero layers and preserves fixed
template properties, including isotopes, roughness, and porosity. The element
list must include every element contributing to total layer thickness.

## Pileup reversal

`spectrum::removePileup(measured, realTime, liveTime, fudgeFactor)` reverses the
native `pileup()` formula, including its live/real-time count scaling. All time
arguments are in seconds; real and live times must be positive. For example:

```cpp
#include <ibeamlab/spectrum_processing.h>

auto measured = ibeamlab::spectrum::pileup(original, 10.0, 8.0, 0.4e-6);
auto restored = ibeamlab::spectrum::removePileup(measured, 10.0, 8.0, 0.4e-6);
```

Supply the complete `2*N-1` output channels, including the pileup tail, to
recover `N` original channels. The routine first solves for the original count
total, then reconstructs channels sequentially and checks the complete spectrum
against the forward formula. It rejects invalid timing, nonfinite/negative
counts, even lengths, and spectra inconsistent with the model. The optional
`relativeTolerance` defaults to `1e-6`, measured against the peak input count,
to accommodate float32 rounding. Tiny negative reconstruction roundoff is
clamped to zero. Zero pileup still reverses live/real-time scaling.

This is a model inversion rather than a statistical fit to noisy measurements.
Cropped or inconsistent tails are unsupported; loss of tail information cannot
be repaired by zero padding. A passing check establishes consistency with this
model, not that the supplied data were recorded without truncation. Channel
reconstruction takes O(N^2) work. Python exposes the same function as
`ibeamlab.spectrum.remove_pileup`.
