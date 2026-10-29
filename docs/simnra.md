# SIMNRA integration

Enable `IBEAMLAB_ENABLE_SIMNRA=ON` in the root CMake project on Windows.
SIMNRA must be installed and registered for COM. Each measurement maps its
label and particle to a reference XNRA file, which supplies geometry and
cross-section choices. Measurements may have different beam energies.

`SimnraConfig.multithreaded_apartment` and the Python wrapper default to `true`.
The data-generation TOML exposes the same setting under `[simulation]`.
Worker sessions use MTA and retain their reference settings. Batch simulation
uses the configured worker pool while preserving result order; core dataset
generation submits bounded batches to this same worker pool. Batch size is
independent of dataset shard size; it should be at least the worker count.

`fast_calculation=true` explicitly selects `CalculateSpectrumFast`, which can
produce different numerical spectra. The standalone C++/Python simulator defaults
to standard calculation. The data-generation example selects fast mode explicitly
to match the reference generator. Both modes are checked against independent
calculations using the corresponding algorithm, with unchanged tolerances.
Only total spectra are returned; extra elemental spectra are disabled.

When rebuilding a target, the adapter appends and populates new layers before
removing the old prefix. This preserves cross-section choices. Compatible
layer topology is reused; `reuse_target_topology=false` forces rebuilding for
diagnostics. Pileup follows the reference: `nopu` means no pileup.

## Live verification

On the local Windows machine with SIMNRA installed:

```console
cmake -S . -B build/simnra -DIBEAMLAB_BUILD_PYTHON=ON -DIBEAMLAB_ENABLE_SIMNRA=ON -DIBEAMLAB_REQUIRE_SIMNRA_TEST_RUNTIME=ON
cmake --build build/simnra --config Release --parallel
ctest --test-dir build/simnra -C Release -L simnra --no-tests=error --output-on-failure
```

Use a Python environment with NumPy, pybind11 and pywin32. The suite checks all
five `xnra/` references against independent fresh SIMNRA calculations, two
workers, repeated sessions, forced append-before-delete rebuilds and a 10%
first-layer thickness change. Baseline tolerance is `rtol=1e-5, atol=1e-5`;
worker repeatability uses `rtol=1e-10, atol=1e-10`.

The 2026-10-29 MTA run passed all five references on SIMNRA 7.04, with a maximum
fresh-baseline difference of approximately `2.27e-13` counts. A new run writes
`simnra-xnra-results.json` at the build-directory root.

Saved spectra can be outdated when a project was saved without recalculation.
Stored-curve mismatches are diagnostics by default; pass/fail uses fresh
independent calculations. `IBEAMLAB_CHECK_SAVED_SIMNRA_SPECTRA=ON` enables the
additional strict stored-curve check.

The `cpp/tests/simnra_precision_probe.py` script investigates concentration
round-trip sensitivity. Run COM diagnostics separately from other SIMNRA tests.
[Validation history](validation-history.md) retains the earlier investigations.
GitHub runners compile the adapter but exclude licensed execution.
