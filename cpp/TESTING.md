# Implemented tests and CI

This inventory covers the standalone `cpp/` project. CTest counts test programs
or suites, not individual assertions. Some suites contain several scenarios.

| CTest name | Labels | Implemented coverage |
| --- | --- | --- |
| `core` | unit | Valid/invalid experiments, independent measurement energies, parameter reads/writes, coupled composition updates, rollback, duplicate/unknown targets, nonfinite values, bounds, validation paths, units |
| `processing` | unit | Calibration edges, rebin identity/conservation/partial coverage, invalid grids and aliasing, FFT pileup versus direct convolution, inverse round trip, zero counts and invalid acquisition settings |
| `generation` | integration | Dummy simulation ? shards ? streaming reads; row order, provenance, rewind, overwrite rejection, record/discard/stop policies, cancellation, incomplete/corrupt data rejection and seeded sampling |
| `packages` | integration | Metadata and binary package round trip, overwrite/corruption/normalization rejection; with ONNX enabled: forward/inverse numerical predictions, batch parity, prepared-input parity, ordered scaler/custom pipelines, per-measurement exposure/rebin/pileup corrections, unequal-length multi-spectrum packing and tensor schema rejection |
| `installed-consumer` | integration, deployment | Install the library, configure/build an independent C++ consumer with `find_package`, then run it |
| `python-api` | integration, python | Python domain API, dummy simulation, caller-owned NumPy buffers, dtype rejection, generation callbacks, dataset reads; with ONNX: multi-measurement deployment and PyTorch export parity at two batch sizes |
| `allocation-benchmark` | performance | 100 rebin+pileup iterations at 5,000 channels; with ONNX, 100 single-detector predictions into reused output at 5,000 channels, checking no counted C++ allocations after warmup |
| `simnra-reference` | integration, licensed, simnra | Five repository XNRA projects; independent fresh COM calculations, two workers, session reuse, forced append-before-delete rebuilds, a 10% layer-thickness change and pileup flags |

`core` can be disabled with `IBEAMLAB_CORE_BUILD_TESTS=OFF`. Python, allocation,
and licensed suites are registered when their corresponding build options are
enabled. ONNX-disabled package tests still run serialization/validation checks
and verify that inference reports its unavailability.

## Run locally

```console
cmake -S cpp -B cpp/build -DIBEAMLAB_BUILD_TESTS=ON
cmake --build cpp/build --config Release --parallel
ctest --test-dir cpp/build -C Release -L unit --no-tests=error --output-on-failure
ctest --test-dir cpp/build -C Release -L integration -LE licensed --no-tests=error --output-on-failure
```

Enable `IBEAMLAB_BUILD_PYTHON=ON` for Python integration and install NumPy/pybind11
in the selected Python environment. ONNX-enabled integration also needs the C++
SDK, PyTorch and ONNX. Enable `IBEAMLAB_BUILD_BENCHMARKS=ON` for the performance
label. The benchmark does not enforce a machine-dependent timing threshold.
The inference allocation check intercepts C++ `new` in the executable and, for
static builds, the linked library. It does not measure total heap use or all
ONNX Runtime allocations, and cannot intercept allocations in a Windows DLL. Use the static build
for the library allocation regression; shared builds also check numerical output.
Forward contract tests cover every parameter target type, immutable fields,
invalid physical values, caller-owned output reuse, and changing measurement order.

## GitHub Actions

`.github/workflows/cpp-tests.yml` runs on pushes/pull requests changing the
C++ project, XNRA references, or CI configuration. It can also be run manually.

- Windows and Ubuntu, static and shared: unit tests, integration tests including
  Python and installed consumers, and allocation checks. ONNX/SIMNRA are disabled.
- Ubuntu Debug: ASan and UBSan on unit/integration tests, including installed consumption.
- Windows, static and shared, ONNX Runtime 1.22.1: unit and integration checks,
  including the real inference graphs, PyTorch export parity and allocation checks.
  The Windows SIMNRA adapter is also compile-checked; execution tests labelled
  `licensed` are excluded because GitHub runners do not have SIMNRA installed.

Each job uploads JUnit XML and CTest logs even when tests fail. CI uses the dummy
simulator for simulation/generation integration. Real SIMNRA reference, worker
and cross-section preservation tests remain local-only on a machine with the
licensed COM application. The workflow syntax is checked with actionlint;
hosted Linux/Windows jobs still need execution in GitHub Actions.

## Saved XNRA spectra

The stored curves may be outdated because a project can be saved without
recalculating. They are recorded as diagnostics by default; the licensed test's
pass/fail result uses fresh independent calculations and behavioral checks.
This does not alter reference files or relax numerical output tolerances.

For a strict audit of stored curves, enable
`IBEAMLAB_CHECK_SAVED_SIMNRA_SPECTRA=ON` when configuring a local build. This can fail for the
known outdated RBS no-pileup curve until the project is recalculated and saved.
The script itself also accepts `--check-saved-spectra`.

The independent `tests/simnra_precision_probe.py` diagnostic reproduces the
composition serialization sensitivity discussed in `SIMNRA_VALIDATION.md`.
Run standalone SIMNRA diagnostics separately from the integration suite.
