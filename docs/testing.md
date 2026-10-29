# Implemented tests and CI

This inventory covers the unified root project. CTest counts test programs
or suites, not individual assertions. Some suites contain several scenarios.

| CTest name | Labels | Implemented coverage |
| --- | --- | --- |
| `core` | unit | Valid/invalid experiments, independent measurement energies, parameter reads/writes, coupled composition updates, rollback, duplicate/unknown targets, nonfinite values, bounds, validation paths, units |
| `processing` | unit | Calibration edges, rebin identity/conservation/partial coverage, invalid grids and aliasing, FFT pileup versus direct convolution, inverse round trip, zero counts and invalid acquisition settings |
| `generation` | integration | Dummy simulation ? shards ? streaming reads; row order, provenance, rewind, overwrite rejection, record/discard/stop policies, cancellation, incomplete/corrupt data rejection and seeded sampling |
| `packages` | integration | Metadata and binary package round trip, overwrite/corruption/normalization rejection; with ONNX enabled: forward/inverse numerical predictions, batch parity, prepared-input parity, ordered scaler/custom pipelines, per-measurement exposure/rebin/pileup corrections, unequal-length multi-spectrum packing and tensor schema rejection |
| `installed-consumer` | integration, deployment | Install the library, configure/build an independent C++ consumer with `find_package`, then run it |
| `python-api` | integration, python | Python domain API, dummy simulation, caller-owned NumPy buffers, dtype rejection, generation callbacks, dataset reads; with ONNX: multi-measurement deployment and PyTorch export parity at two batch sizes |
| `python-wrapper` | integration, python | Root Python package, independent measurement settings, cached experiment conversion, NumPy result lifetime/read-only views, buffer reuse, streaming datasets, failure/cancellation policies; with ONNX: multi-detector metadata/inference, optional training and export |
| `python-data-generation` | integration, python | Multilayer configuration validation without an extension, exact method budgets, compounds/shared 2–8 element sets/independent layerwise 1–13 element sets, reproducibility, pure samples, parameter columns, dummy generation and selected-element audit artifacts |
| `allocation-benchmark` | performance | 100 rebin+pileup iterations at 5,000 channels; with ONNX, 100 single-detector predictions into reused output at 5,000 channels, checking no counted C++ allocations after warmup |
| `simnra-reference` | integration, licensed, simnra | Five repository XNRA projects; independent fresh COM calculations, two workers, session reuse, forced append-before-delete rebuilds, a 10% layer-thickness change and pileup flags |

`core` can be disabled with `IBEAMLAB_CORE_BUILD_TESTS=OFF`. Python, allocation,
and licensed suites are registered when their corresponding build options are
enabled. ONNX-disabled package tests still run serialization/validation checks
and verify that inference reports its unavailability.

## Run locally

```console
cmake -S . -B build/local -DIBEAMLAB_BUILD_TESTS=ON
cmake --build build/local --config Release --parallel
ctest --test-dir build/local -C Release -L unit --no-tests=error --output-on-failure
ctest --test-dir build/local -C Release -L integration -LE licensed --no-tests=error --output-on-failure
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

[`.github/workflows/cpp-tests.yml`](../.github/workflows/cpp-tests.yml) runs on pushes/pull requests changing the
Root/C++ build files, documentation, Python package/tests, examples, packaging,
XNRA references, or CI configuration.
It can also be run manually.

- Windows and Ubuntu, static and shared: unit tests, integration tests including
  Python and installed consumers, and allocation checks. ONNX/SIMNRA are disabled.
- Windows and Ubuntu: build a Python wheel from the repository root, install it,
  and run `python -I tests/installed_package.py` to prevent source imports from
  masking installation or shared-library loading problems.
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

The independent `cpp/tests/simnra_precision_probe.py` diagnostic reproduces the
composition serialization sensitivity discussed in [SIMNRA notes](simnra.md).
Run standalone SIMNRA diagnostics separately from the integration suite.

## Live generation throughput

The manual benchmark exercises the actual sampled generation path, including
dataset writing, with separate cold warmup and timed intervals:

```console
python cpp/benchmarks/simnra_generation.py --extension-dir build/local/Release/bin --workers 64 --samples 256 --layers 1 --report build/local/Release/simnra-throughput.json
```

It uses temporary datasets and compares a serial standard-mode run with bounded
parallel fast-mode generation. Run it separately from other COM diagnostics.
Use `--layers 8` to measure the thicker multilayer system. Fast mode is a distinct
numerical algorithm; the live reference suite compares it against independent
Fast calculations as well as retaining the standard-mode comparisons.
