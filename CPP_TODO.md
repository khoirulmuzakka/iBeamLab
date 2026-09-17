# iBeamLab implementation responsibilities and status

This checklist is the project boundary. Python owns experiment design and
training. C++ owns the stable simulator, storage, package, and deployment
runtime shared with AutoNRA.

## Responsibility boundary

### Python

- [x] Construct fixed and open parameter configurations.
- [x] Sample open parameter matrices in Python.
- [x] Normalize sampled concentrations independently per layer.
- [x] Pass complete parameter matrices to the native generator.
- [x] Record Python sampler name, version, seed, and TOML configuration in the
  native dataset manifest.
- [x] Add the variable-layer example: one dataset per layer count, controlled
  entirely by Python.
- [x] Keep its thickness and concentration sampling policies transparent in
  Python.
- [ ] Add Python dataset collection/loading utilities, parameter padding, and
  layer masks.
- [ ] Add train/validation/test splitting and optional augmentation/noise.
- [ ] Add PyTorch models, losses, training, evaluation, and ONNX export.
- [ ] Build model metadata from the training configuration and call the native
  model-package writer.

### C++

- [x] Provide the backend-neutral simulator abstraction.
- [x] Implement SIMNRA COM lifecycle, reference-file isolation, reusable
  thread-owned workers, parallel simulation, cancellation, and failures.
- [x] Validate and materialize Python-supplied parameter rows.
- [x] Read and write datasets, spectra, failures, metadata, and provenance.
- [x] Apply packaged production preprocessing consistently.
- [x] Read and validate directory and ZIP model packages.
- [x] Run explicit inverse inference (spectra to sample parameters) and forward
  inference (sample/setup to labeled spectra).
- [x] Write/dump directory and ZIP model packages from model bytes and metadata.
- [x] Expose the native model-package writer to Python.

Sampling distributions, variable-layer policies, dataset splitting, model
definitions, and training must not be added to the native library.

## Completed

- [x] C++20 CMake project with Debug/Release support and consistent outputs.
- [x] One installable/exported native target named `ibeamlab`.
- [x] Optional Python, SIMNRA, ONNX Runtime, tests, examples, benchmarks, and
  sanitizer switches.
- [x] Clean build outputs from `compile.bat`: the C++ import library in `lib/`
  and the Python extension plus runtime DLLs beside `ibeamlab/__init__.py`,
  without a `Release/` subdirectory.
- [x] One integrated SIMNRA implementation and one integrated pileup/rebin
  implementation; obsolete standalone modules and bindings removed.
- [x] Dependency-light sample, isotope, species, layer, beam, detector, and
  experimental-setup value types with units, validation, stable ordering, and
  versioned TOML round-tripping.
- [x] Backend-neutral batch simulator API with float32 spectra, structured
  per-item failures, cooperative cancellation, and deterministic dummy backend.
- [x] Persistent SIMNRA workers with thread-owned COM instances, one instance per
  measurement, reference-file reuse, cached target topology, per-request value
  updates, total and optional elemental spectra, idempotent shutdown, and UTF-8
  error conversion.
- [x] Fixed/open generation parameters for layer, concentration, beam, detector,
  calibration, resolution, and particles-per-steradian values.
- [x] Validated parameter materialization; sampling policy is supplied by Python.
- [x] Streaming sharded dataset writer/reader with bounded record sizes, sample
  IDs, parameters, labeled spectra, failures, invalid counters, transactional
  TOML manifest updates, incomplete-state detection, provenance, and corruption
  rejection.
- [x] Bounded native data generation with batch sizing, Stop/Record/Discard
  policies, progress callbacks, cancellation, and a headless CLI.
- [x] Preprocessing for selection/order, crop/pad, concatenation, clipping,
  rebinning, pileup, constant/standard/min-max/log transforms, pipelines, and
  inverse output transforms.
- [x] Safe `.ibeam.zip`/directory model-package reader using `package.toml`, exact
  entry selection, path rejection, size limits, ZIP CRC validation, declared
  model size/CRC32 checks, method sizes, named features, units, and preprocessing.
- [x] Long-lived ONNX Runtime inference session with PImpl, float32 name/type/shape
  validation, single/batch inference, labeled spectrum flattening, fitted
  preprocessing, inverse output transforms, named unit-bearing results, CPU thread
  controls, and an extensible provider option.
- [x] Unified `_ibeamlab_cpp` module linked to the single `ibeamlab` library, with
  logical submodules, NumPy spectrum/matrix
  exchange, GIL release around native work, progress callbacks, and cancellation.
- [x] Native and Python smoke tests covering numerical operations, invalid inputs,
  deterministic generation, incomplete/corrupt datasets, package validation,
  real ONNX inference from directory/ZIP, repeated construction, and shutdown.
- [x] Native CLI/example, consumer `find_package` project, benchmark target,
  formatting/static-analysis configuration, documentation, MIT project license,
  and third-party license notes.

## Environment-gated verification

- [x] The installed-SIMNRA integration test is implemented and is enabled when
  `IBEAMLAB_SIMNRA_REFERENCE` points to a usable reference file.
- [ ] Run that integration test and record numerical reference tolerances on a
  machine with the licensed SIMNRA COM application and suitable reference files.

## Explicitly postponed

- Python model definitions and PyTorch training loops.
- Hyperparameter optimization.
- AutoNRA adapters or build integration.
- GUI and distributed generation.
- GPU execution-provider tuning.
