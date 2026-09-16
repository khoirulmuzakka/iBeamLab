# iBeamLab native C++ implementation status

The native foundation is implemented. Python model definitions, PyTorch training,
Python package export, AutoNRA integration, and GUI work remain intentionally
postponed.

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
- [x] Deterministic versioned uniform sampling and validated materialization.
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
- Python ONNX/TOML package writer.
- AutoNRA adapters or build integration.
- GUI and distributed generation.
- GPU execution-provider tuning.
