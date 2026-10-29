# C++ development TODO

Implement the new library in `cpp/` using its own CMake project. Use `native/`
as reference only. Python bindings, examples, and documentation are included in this implementation.
Keep module boundaries clear; separate DLLs are not currently required.

## 0. Core foundation — implemented

- [x] Sample, layer, species, and isotope value types.
- [x] Independent measurements with beam, detector, and acquisition settings.
- [x] Typed parameter targets and explicit read/write functions.
- [x] Validated atomic parameter updates and finite inclusive bounds.
- [x] Unit conventions and validation issues with paths.
- [x] Standalone CMake project and passing core tests.

## 1. Spectrum representation and processing — next

- [x] Define channel-center/edge conventions, count units, and calibration rules.
- [x] Choose count precision and accumulation precision explicitly.
- [x] Add `core/spectrum.h`: owning contiguous counts and a measurement label.
- [x] Add non-owning spectrum views with explicit lifetime requirements.
- [x] Define validation for counts, calibration, and channel-grid coverage.
- [x] Implement count-preserving rebinning in `processing`.
- [x] Support caller-owned output buffers and convenient owning results.
- [x] Specify output sizes, overlap rules, and out-of-range behavior.
- [x] Verify identity rebinning, count conservation, partial overlap, and invalid grids.
- [x] Settle pileup model, acquisition-time conventions, and correction ordering.
- [x] Implement pileup application/removal with reusable scratch storage.
- [x] Add numerical reference checks for pileup and its supported inverse regime.

## 2. Simulation

- [x] Define backend-independent single/batch simulation interfaces.
- [x] Define labeled results, per-measurement failures, cancellation, and progress.
- [x] State thread-safety, worker ownership, and output-buffer contracts.
- [x] Implement deterministic dummy backend for workflow tests.
- [x] Implement SIMNRA backend, isolated behind a Windows build option.
- [x] Implement COM lifecycle, reference-file isolation, and reusable workers.
- [x] Verify ordering, failures, cancellation, repeated use, and shutdown.
- [x] Run physical reference comparisons against the five repository XNRA projects.
- [x] Identify and fix the NRA reference-test round-trip input mismatch.
- [ ] Explain the RBS no-pileup saved-spectrum mismatch recorded in `SIMNRA_VALIDATION.md`.
- [ ] Investigate the observed SIMNRA sensitivity to changing species order.

## 3. Dataset storage

- [x] Specify versioned manifest and record/shard formats.
- [x] Store sample IDs, parameter order, measurement labels, units, and provenance.
- [x] Define representation of successful, failed, and incomplete records.
- [x] Implement bounded-memory writer and streaming reader.
- [x] Validate dimensions, record limits, checksums, and format compatibility.
- [x] Make completion publication transactional and reject unintended overwrites.
- [x] Verify round trips, corruption, interrupted writes, and variable spectrum lengths.

## 4. Data generation

- [x] Define the generation plan and ordered parameter matrix inputs.
- [x] Define coupled composition handling without implicit normalization.
- [x] Keep sampling policy separate from simulation orchestration.
- [x] Apply complete candidate parameter sets and validate before simulation.
- [x] Stream bounded batches from simulator to dataset writer.
- [x] Implement stop/record/discard failure policies and cancellation.
- [x] Record seeds, configuration, backend information, and reference fingerprints.
- [x] Verify deterministic dummy runs, parameter ordering, and bounded memory use.

## 5. Model packages and inference

- [x] Define shared, versioned package metadata and scientific input/output schemas.
- [x] Specify preprocessing, units, element order, and reference measurement settings.
- [x] Implement validated package reading/writing in `model`.
- [x] Implement private reusable ONNX execution behind an optional dependency.
- [x] Implement `inference::ForwardModel` and `inference::InverseModel`.
- [x] Specify correction ordering and reject unsupported measurement changes.
- [x] Define inverse output and optional uncertainty semantics.
- [x] Verify package validation, batch shapes, and numerical reference predictions.
- [x] Verify core, simulation, and datasets build without ONNX.

## 6. Fitting

- [ ] Define fit parameters, initial values, bounds, scaling, and coupled constraints.
- [ ] Define forward-evaluation adapters for simulation and forward inference.
- [ ] Define objectives appropriate to count data and optional residual weighting.
- [ ] Choose optimizer dependencies and convergence criteria.
- [ ] Implement fit results, diagnostics, cancellation, and failure handling.
- [ ] Allow optional inverse-model initialization without requiring an inverse model.
- [ ] Verify recovery on known synthetic cases and behavior for ambiguous solutions.

## 7. Performance and build verification

- [x] Keep spectra contiguous; avoid unnecessary owning copies in repeated operations.
- [x] Reuse output and scratch buffers; specify which APIs may allocate.
- [x] Benchmark representative spectra of approximately 5,000 channels.
- [x] Keep compiled parameter access and additional processing parallelism deferred;
  the current allocation benchmark does not justify adding either.
- [x] Extend CMake and tests as each module is implemented.
- [x] Verify supported static/shared configurations and optional-backend builds.
- [x] Add sanitizer checks where supported.
- [x] Validate exported CMake targets and independent C++ consumption before release.

## Integration

- [x] Python bindings and wrapper updates.
- [x] Training/export integration with the new package contract.
- [x] Examples and user documentation.
- [x] Automated clean-build and test workflows.

## Verification notes

- Static and backend-free Windows builds have passing core/workflow/installed-consumer checks.
- ONNX reference graphs and Python/PyTorch export parity checks are implemented.
- The 5,000-channel rebin+pileup benchmark verifies zero allocations after warmup.
- Linux sanitizer/clean-runner jobs are configured in CI; execution is pending there.
- Licensed SIMNRA verification now runs directly against `xnra/`; all five backend cases match fresh
  calculations; one saved-spectrum comparison remains failed. See `SIMNRA_VALIDATION.md`.
- iBeamLab datasets/packages have their own format 1; legacy formats and existing
  notebooks are not automatically migrated. The new exporter integrates custom
  training models with the new contract.
