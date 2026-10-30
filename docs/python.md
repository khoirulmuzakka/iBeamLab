# Python interface

The `ibeamlab/` package uses the current C++ library in `cpp/`. Its physical
objects follow the same sample, measurement and experimental-setup model.
The wrapper provides simulation, generation, streaming datasets, optional
PyTorch training, format 1 model export and forward/inverse inference.

## Build and dependencies

See [building.md](building.md) for installation and optional ONNX/SIMNRA builds.
For a source-tree build, add its bin directory (for example, build/local/bin)
to PYTHONPATH. The extension and PyTorch are loaded only when used.
When using `compile.bat Release`, the extension directory is
`build/local/Release/bin`.

## Measurements and simulation

```python
import ibeamlab as ibl

experiment = ibl.Experiment(
    ibl.Sample([ibl.Layer(100, {"Si": 0.5, "O": 0.5})]),
    ibl.ExperimentalSetup([
        ibl.Measurement("RBS", ibl.Beam("H", 2000),
                        acquisition=ibl.Acquisition(1000, 10, 8)),
        ibl.Measurement("NRA", ibl.Beam("H", 3000),
                        acquisition=ibl.Acquisition(2000, 10, 8)),
    ]),
)
result = ibl.DummySimulator(channels=5000).simulate(experiment)
counts = result["RBS"].counts  # contiguous float64 NumPy array
```

Energies are keV, time is seconds, and thickness is `1e15 atoms/cm2`. Beams may
differ between measurements. Physical value objects are frozen; use
`dataclasses.replace` to construct variations. Experiments cache their C++
representation on first use. Treat `to_core()` as read-only; use
`ibeamlab.native` for mutable C++ workflows.

For SIMNRA, construct `SimnraMethod("RBS", "reference.xnra", "H")` for each
measurement and pass the list to `SimnraSimulator`. Use it as a context manager
to release COM sessions promptly. Result failures are available through
`result.failure` and `result.failed`.

## Generate and read data

```python
plan = ibl.GenerationPlan(experiment, [
    ibl.vary.layer_thickness(layer=0, bounds=(50, 150)),
    ibl.vary.beam_energy("RBS", bounds=(1900, 2100)),
])
rows = plan.sample_uniform(100, seed=42)
summary = plan.generate(rows, ibl.DummySimulator(), "generated-data")
dataset = ibl.open_dataset("generated-data")
for batch in dataset.iter_batches(32):
    pass  # bounded-memory iteration; each record has parameters and result

inputs = dataset.parameters       # successful rows only, physical float64
targets = dataset.spectra("RBS")  # same successful row order
```

Generation never implicitly normalizes concentrations. Supply coupled
composition rows yourself. Bounds affect sampling; explicit generation rows
are checked for physical validity by the core. Destinations must not already
exist. `FailurePolicy.RECORD` retains failures, `DISCARD` omits them, and `STOP`
leaves an incomplete dataset on failure. Cancellation uses
`simulator.request_stop()`; incomplete datasets cannot be opened as complete.
Progress callbacks receive `(completed, total)`.

Dataset iteration includes both successful and failed stored records; discarded
rows have no record. Each iteration opens an independent streaming cursor.
Matrix helpers allocate on demand and include successful rows only. Variable
spectrum lengths require record iteration rather than stacking.

See [DataGeneration](../examples/DataGeneration/README.md) for a complete
TOML-driven multilayer example with regional and compound sampling.

## Metadata and inference

CPU and CUDA execution are session options, independent of package metadata:

```python
options = ibl.InferenceOptions(execution_provider="cuda", device_id=0,
    intra_op_threads=1, inter_op_threads=1, enable_graph_optimizations=True)
model = ibl.load_model("forward.ibm", options=options)  # Also accepts inverse packages.
print(model.options)
print(ibl.available_execution_providers())
```

The default provider is `cpu`. CUDA requires a CUDA-enabled native build and
working runtime dependencies. An unavailable CUDA provider or invalid GPU ID
raises an error; it does not silently create a CPU-only session. CPU shape/control
operators may participate in CUDA graphs. Native inference does not require
PyTorch or Python ONNX Runtime. `allow_tf32=False` is the default; explicitly
enable it to permit reduced-mantissa tensor-core arithmetic after checking accuracy.
Host input/output buffers and tensor handles are reused where addresses/shapes
remain stable. CUDA transfers use ONNX Runtime's device arena and complete before
physical CPU postprocessing begins. `native.run_graph` also accepts native
`InferenceOptions` as its third argument for raw batched graph execution.

Metadata separates physical measurement preparation from numerical training
transforms. The four stages execute as input_preparation, input_transform,
ONNX, output_transform, output_preparation. Forward input preparation packs
physical sample parameters or elemental AD; inverse output preparation is an
`EdpSchema` with ordered elements and a fixed maximum layer count.

```python
schema = ibl.SpectraSchema([
    ibl.SpectrumSpec("RBS", 2998, state=ibl.SpectrumState.BARE),
    ibl.SpectrumSpec("NRA", 3000, state=ibl.SpectrumState.BARE),
])
forward = ibl.ForwardModelMetadata(
    reference_setup=experiment.setup,
    input_preparation=ibl.LayerInputSchema(["Si", "O"]),
    output_spectra=schema,
    input_transform=ibl.TransformPipeline([ibl.MultiplyScale([1e-4])]),
    output_transform=[
        ibl.SpectrumTransform("RBS", ibl.TransformPipeline(
            [ibl.MultiplyScale([.01])], direction=ibl.TransformDirection.INVERSE)),
        ibl.SpectrumTransform("NRA", ibl.TransformPipeline(
            [ibl.StandardScale([0], [25])], direction=ibl.TransformDirection.INVERSE)),
    ],
    output_preparation=[
        ibl.SpectrumProcessing("RBS", [ibl.Rebin(), ibl.ExposureScale(), ibl.Pileup(4e-7)]),
        ibl.SpectrumProcessing("NRA", [ibl.Rebin(), ibl.ExposureScale()]),
    ],
)
metadata = ibl.ModelMetadata(forward, provenance="training-run-42")
metadata.validate()
# package = ibl.Package.from_onnx("model.onnx", metadata)
# package.write("forward.ibm")
```

Store the actual fitted training coefficients. `direction=INVERSE` undoes
operations in reverse order; clipped scaling is not invertible. `location=GRAPH`
declares a transform already embedded in ONNX; the default RUNTIME executes it
outside ONNX. Exporters must declare the graph boundary accurately. A transform
is never fitted or applied in both places by the runtime.

Forward layer graphs accept `[batch,layers,elements]` and have no layer maximum.
`predict(experiment)` packs physical AD; `predict_areal_density(ad)` accepts
contiguous float32 `[layers,elements]`. Runtime transforms skip padding rows.
Each output spectrum has its own numerical transform and physical corrections.

Inverse models use per-detector `input_transform` pipelines after physical
`input_preparation`. Their `output_transform` restores physical EDP units before
`output_preparation=ibl.EdpSchema(elements, layers, uncertainty)` decodes arrays.
With uncertainty, only affine numerical transforms are supported: std is scaled
by absolute scale and probabilities are unchanged.

Reference-condition inference keeps numerical transforms while bypassing physical
measurement corrections:

```python
reference_spectra = forward_model.predict_reference(experiment)
reference_population = forward_model.predict_reference_areal_density_batch(ad_population)
edp = inverse_model.predict_reference({"RBS": reference_counts})
edps = inverse_model.predict_reference_batch(reference_measurements)
```

Use `predict` for measured/requested conditions, and `predict_reference` for
physical inputs/outputs at the packaged reference response. All batch variants
execute one ONNX call. Raw `native.run_graph` expects graph-boundary values and
skips all runtime stages. Layer export still requires an architecture-specific
dynamic `graph_exporter` callback; symbolic axis names do not make a traced loop
dynamic. Model architectures remain application/example code.

The surrogate notebook exposes training scalers separately from export-time
pileup controls. Its output preparation uses rebinning, exposure scaling and
optional requested-time pileup. The full result may have `2*N-1` channels.
`InferenceOptions(apply_pileup=False)` remains a selective override of declared
forward pileup steps; prefer reference prediction when all physical corrections
must be bypassed together.

For differential evolution, evaluate the entire candidate population in one
ONNX call, without constructing sample objects:

```python
# Physical AD, not pre-scaled: [population,layers,elements], metadata element order.
ad = np.ascontiguousarray(candidate_areal_densities, dtype=np.float32)
# Sum of processed detector widths; 2998 is an example for one RBS detector.
out = np.empty((len(ad), 2998), dtype=np.float64)
spectra = model.predict_areal_density_batch(ad, setup=requested_setup, out=out)
rbs = spectra["RBS"]  # [population,processed_channels], a view into out.
```

For population reduction, allocate input/output arrays for the maximum population
and pass contiguous active slices such as `ad[:n]` and `out[:n]`. Native buffers
retain their capacity when the population shrinks. ONNX tensor wrappers are
recreated when shapes change; this does not copy their backing arrays and does
not guarantee allocation-free execution inside ONNX Runtime.

Reuse `out` for subsequent iterations. Returned
arrays change when that storage is overwritten; omit `out` for independently
owned results. An empty preprocessing pipeline borrows compatible float32 AD;
otherwise the model retains its preprocessing scratch. The graph output and
detector-processing workspaces are also retained. Python dictionary/view objects
are still created per call; ONNX Runtime manages its own internal allocations.
`setup` applies to every candidate in this population. Different fitting workers
can each own their own model with CPU or CUDA options and run concurrently.

`ForwardModel.predict_batch(experiments)` also executes one graph call, padding
layer inputs to the batch's maximum layer count and applying each experiment's
requested detector processing afterward. `InverseModel.predict_batch(inputs)`
prepares inputs independently, then executes one graph call. Empty batches return
empty results without running ONNX. A graph with a fixed batch axis must receive
that exact batch size; unsupported sizes raise an error rather than running a
hidden loop over individual predictions. Model loading releases the Python GIL,
as do graph execution and native population processing.

Inverse metadata defines `reference_setup`, `input_spectra` and a fixed-layer
`EdpSchema` in `output_preparation`. Physical `input_preparation` and numerical
`input_transform` are independently configured per detector. `output_transform`
restores physical elemental areal densities after the graph.

Inverse input uses `InverseInput(setup, {"RBS": counts, "NRA": other_counts})`.
Spectra are matched by label independently of caller order. Predictions expose
`values`, `presence_probability` and `posterior_std` as `[layer, element]` arrays.
`predict_reference` skips physical preparation while retaining tensor and EDP
processing. Model instances must not be called concurrently; use one per worker.
`Package.metadata` is a snapshot; assign an edited snapshot back to update it.

## Optional training and export

`ibeamlab.training.fit` trains a user-supplied CPU PyTorch model with optional
validation data, optimizer and loss. Inputs and targets must already be in
training units. It returns epoch losses and restores the model's original
training/evaluation mode. Architecture selection remains with the caller.

```python
from ibeamlab.training import fit
# history = fit(torch_model, scaled_inputs, scaled_targets,
#               validation_data=(validation_inputs, validation_targets))
# ibl.export_package(torch_model, example_tensor, metadata, "model.ibm")
```

`example_tensor` is preprocessed float32 CPU data shaped `[batch, input_width]`.
Export checks raw PyTorch/ONNX parity at two batch sizes by default, then stores
the graph and processing metadata in format 1. Declare operations outside the
graph only once. Export verification requires ONNX-enabled C++ inference.

## Array ownership and lower-level access

Spectrum arrays use double precision. Existing compatible float64 arrays are
accepted without a Python-side copy; crossing vector-based bindings still
materializes C++ storage. Result arrays are read-only NumPy views that retain
their C++ owner, avoiding intermediate lists and additional count-array copies.
Use `.copy()` if a writable result is required. For
rebinning and pileup, pass reusable `output=` arrays and a `PileupWorkspace` to
avoid repeated Python buffer allocation.

`from ibeamlab import native` exposes the current C++ bindings, including raw
metadata, readers/writers and model execution. It does not use the archived
extension from `old/`. See [test_wrapper.py](../tests/test_wrapper.py) for executable examples and
[testing.md](testing.md) for test commands.
