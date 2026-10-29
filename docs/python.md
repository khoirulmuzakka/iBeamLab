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

Metadata fields have concrete Python/native types and typed ordered pipelines:
`TensorPipeline`, `SpectrumPipeline` and `EdpPipeline`. Native descriptor stubs
ship with the package, while normal imports remain lazy. Wrapper conversion
failures raise `ValidationError` with a field path such as
`LayerInputSchema.preprocessing`; physical consistency checks stay in C++.

```python
schema = ibl.SpectraSchema([
    ibl.SpectrumSpec("RBS", 5000), ibl.SpectrumSpec("NRA", 3000),
])
forward = ibl.ForwardModelMetadata(
    reference_setup=experiment.setup,
    input=ibl.ParameterInputSchema(
        reference_sample=experiment.sample,
        parameters=plan.parameters,
        preprocessing=[ibl.StandardScale([0, 2000], [100, 100])],
    ),
    output_spectra=schema,
    postprocessing=[
        ibl.SpectrumProcessing("RBS", [ibl.ExposureScale(), ibl.Rebin()]),
        ibl.SpectrumProcessing("NRA", [ibl.MultiplyCounts(100), ibl.Rebin()]),
    ],
)
metadata = ibl.ModelMetadata(forward, provenance="training-run-42")
metadata.validate()
# package = ibl.Package.from_onnx("model.onnx", metadata)
# package.write("forward.ibm")
# model = ibl.load_model("forward.ibm")
# predicted = model.predict(experiment)
```

Graph outputs concatenate spectra in schema order; lengths may differ.
For a variable-layer forward model, replace the input with
`ibl.LayerInputSchema(["Si", "O"], preprocessing=[ibl.MultiplyScale([1e-5])])`.
Its graph accepts `[batch,layers,elements]`, and the package stores no layer count.
`model.predict(experiment)` packs physical AD automatically;
`model.predict_areal_density(ad)` accepts physical `[layers,elements]` arrays
directly. Contiguous float32 inputs avoid a Python copy; an optional `setup=`
selects requested acquisition/calibration settings. Raw `run_graph` bypasses
physical preprocessing and expects graph-ready, already-scaled values.
Layer export requires an architecture-specific `graph_exporter` callback to
`export_package`; applications choose how to represent dynamic layer computation.
A symbolic dimension alone does not make a traced Python loop dynamic. The
framework does not select, construct or train a particular network architecture.
Each forward output has its own labelled physical processing pipeline.
Inverse metadata similarly defines `reference_setup`, `input_spectra`, elements
and layers. Its `input_preparation` applies physical operations per measurement;
its current `preprocessing` applies to the complete packed tensor. Independent
per-spectrum tensor-transform sequences are not yet implemented in the core.

Inverse input uses `InverseInput(setup, {"RBS": counts, "NRA": other_counts})`.
Spectra are matched by label independently of caller order. Predictions expose
`values`, `presence_probability` and `posterior_std` as `[layer, element]` arrays.
`predict_prepared` skips physical preparation while retaining tensor and EDP
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
