# C++ API and scientific contracts

Build and install through the root project; see [building.md](building.md).
Public headers live under cpp/include/ibeamlab and document units and ownership.
Consumers link the exported target ibeamlab::ibeamlab.

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

Model metadata declares four stages: `input_preparation`, `input_transform`,
`output_transform` and `output_preparation`. Preparation describes physical
packing or detector operations; transforms describe fitted numerical training
operations. Each transform explicitly declares its direction and whether it
executes in the graph or runtime. Forward output transforms and inverse input
transforms are independent per detector. See [model-format.md](model-format.md).

`predict_reference` and `predict_reference_batch` skip physical detector
corrections while retaining numerical transforms. Forward models also expose
`predict_reference_areal_density` and `predict_reference_areal_density_batch`.
These APIs return spectra at the package reference setup; ordinary prediction
applies the requested measurement corrections. Inverse reference prediction
accepts spectra already at the declared reference setup.

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

Forward metadata separates `ParameterInputSchema` (fixed sample/parameters) from
`LayerInputSchema` (ordered elemental AD and any positive layer count). For the
layer schema, `ForwardModel::predict(experiment, output)` packs AD into a reused
buffer. `predict_areal_density(span<const float>, layers, output, setup)` accepts
contiguous physical AD directly; setup defaults to the package reference.
The runtime retains element lookups, setup bindings, preprocessing scratch and
output capacity. Repeated calls at or below warmed layer capacity do not allocate
new C++ buffers. Empty preprocessing borrows direct AD without an input copy.
Use the overloads taking caller-owned output to reuse spectrum storage; the
convenience return-by-value overloads create owning results. Instances are not
concurrently callable. ONNX Runtime may manage its own internal allocations.

`predict_batch(inputs, outputs)` executes one ONNX batch and reuses caller-owned
result storage. Layer inputs are padded to the largest layer count in the batch.
Each experiment retains its own detector processing. Inverse batches likewise
prepare each measured input before one graph call.

For a DE population, `predict_areal_density_batch(ad, population, layers, output,
setup)` accepts contiguous physical float32 `[population,layers,elements]` and
writes processed float64 `[population,total_channels]` directly into caller-owned
storage. `spectrum_channels()` returns each processed detector width in schema
order; sum them for `total_channels`. The setup defaults to the reference and
applies to the whole population. AD and output storage must not overlap. This
path avoids allocating experiment/result objects for candidates and retains its
preprocessing, graph, and spectrum workspaces between iterations. Empty batches
skip graph execution. Fixed-batch graphs require the exact declared batch size.
For reducing populations, warm up the largest batch and pass active input/output
spans afterward. Native vectors retain capacity as batch size shrinks. Changed
shapes recreate borrowed ONNX tensor wrappers; internal runtime allocations are
outside the C++ buffer-reuse guarantee.

Pileup is the documented pair-sum model in `processing/pileup.h`. It requires
positive acquisition times with live time <= real time and finite
`tau*total/real_time`. Output includes full `2*N-1` support. Removal requires
that full support and verifies reconstructed counts. Truncated or noisy data
may have no supported inverse and are rejected. This model sums channel indices;
its applicability to an instrument must be validated separately.

Generation reuses a working experiment; it never normalizes concentrations
implicitly. Supply coupled composition rows explicitly. A complete dataset is
published only after all requested rows are accepted, recorded as failures, or
discarded. Cancellation and Stop failures leave an incomplete directory. A new
run rejects existing output directories. Caller-supplied rows are currently
resident; generated spectra and writer memory are bounded by one shard.



## Using the library

Inference tuning is shared by forward/inverse models and is not serialized in
scientific model packages:

```cpp
#include <ibeamlab/inference/models.h>
ibeamlab::inference::InferenceOptions options;
options.execution_provider = ibeamlab::inference::ExecutionProvider::Cuda;
options.device_id = 0;
options.intra_op_threads = 1;
options.enable_graph_optimizations = true;
ibeamlab::inference::ForwardModel model(std::move(package), options);
model.predict(experiment, reusable_output);
```

Use `ExecutionProvider::Cpu` for CPU inference. `available_execution_providers()`
reports the build/runtime capabilities; session creation checks the actual GPU
and dependency availability. Invalid/unavailable CUDA requests fail explicitly.
CUDA also executes supported host control operators where needed. FP32 is the
default (`allow_tf32=false`). Tensor wrappers borrow host storage and are reused
until pointer or shape changes; CUDA device storage is managed by ORT's arena.

The [C++ generation example](../cpp/examples/generate.cpp) shows constructing
an experiment, declaring parameter columns and writing a dataset. Build it
with IBEAMLAB_BUILD_EXAMPLES=ON from the root.

See [model-format.md](model-format.md) for the package contract and the public
[metadata header](../cpp/include/ibeamlab/model/metadata.h) for directional schemas
and processing descriptors.

### Isotope compositions

`core::Isotope` stores a positive `massNumber` and a `fraction`. The containing
`Species.element` supplies the atomic identity. Empty `Species.isotopes` means
natural composition. `core::natural_isotopes(element)`, `isotope_data(element)`
and `resolved_isotopes(species)` in `<ibeamlab/core/isotopes.h>` return immutable
views without allocating. Explicit mixtures are validated against the database;
no natural default is invented for elements without a defined mixture.

`model::LayerInputSchema::isotope_compositions` supplies optional
`ElementComposition` assumptions. Packages store resolved mixtures for all input
elements. Inference resolves and caches model assumptions once and compares sample
compositions without temporary vectors, including explicitly supplied natural
mixtures. Direct AD inputs imply the package's isotope assumptions.

### Detector geometry

`core::Detector::geometry` stores `core::Geometry {incidenceAngle, exitAngle,
scatteringAngle}` in degrees. The default `(0,0,180)` represents normal incidence
and exact backscattering. `Geometry::validate()` checks angle ranges and physical
consistency, including non-coplanar and transmission arrangements. Experimental
setup validation includes geometry. Geometry is persisted in datasets and packages.

`IncidenceAngle`, `ExitAngle`, and `ScatteringAngle` are labelled parameter targets.
A parameter-schema forward model can declare geometry features. Fixed geometry
is checked in parameter, layer-AD and inverse inference; physical spectrum
processing cannot change it. Fixed-geometry comparisons allocate no memory.

SIMNRA maps these angles to `Setup.Alpha`, `Setup.Beta`, and `Setup.Theta`, updating
only changed values after initialization. Each worker claims an unowned application
using SIMNRA's documented `OLEUser` protocol before activating related interfaces.
Application/interface activation is serialized; spectrum calculations remain
parallel. Interactive or already-owned instances are skipped.
