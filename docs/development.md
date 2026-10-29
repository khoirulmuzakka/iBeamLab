# Development status

The root CMake project builds the C++ library and optional Python extension.
See [testing.md](testing.md) for the test inventory and [simnra.md](simnra.md)
for current live SIMNRA verification.

## Implemented

- Shared sample, isotope, species and independent measurement value types.
- Typed parameter targets, validation, bounds and atomic parameter updates.
- Contiguous double spectra and non-owning views with documented lifetimes.
- Calibrated rebinning and pair-sum pileup application/removal.
- Dummy and Windows SIMNRA simulation, including reusable COM workers and MTA.
- Append-before-delete target rebuilds preserving cross-section choices.
- Checksummed sharded datasets, streaming readers and explicit completion.
- Physical parameter plans, failure policies, progress and cancellation.
- Reproducible compound, shared-element and independent layerwise sampling.
- Format 1 metadata and standalone ONNX packages.
- Forward multi-spectrum inference and inverse elemental-depth-profile inference.
- Ordered processing pipelines and application-registered custom transforms.
- Cached forward parameter bindings and direct output for empty pipelines.
- Python domain/workflow wrappers, read-only NumPy result views and optional
  custom-model PyTorch training/export.
- Static/shared builds, installed C++ consumers, wheel smoke tests, allocation
  regression checks and CI sanitizer jobs.

## Remaining work

Fitting is not implemented. Its design still needs parameter scaling and
constraints, objectives for count data, optimizer selection, convergence
criteria, diagnostics, cancellation and synthetic recovery tests.

Inverse input preparation supports per-measurement physical operations, while
tensor preprocessing currently applies to the complete packed input. Independent
per-spectrum tensor-transform sequences remain future work.

High-level generation submits bounded backend batches, preserving deterministic
row identity and cancellation safety while using SIMNRA's configured workers.
Further processing parallelism should follow profiling.

Further validation of SIMNRA species-order sensitivity is documented in the
[validation history](validation-history.md). Outdated saved spectra are
diagnostics; fresh independent calculations determine backend correctness.
