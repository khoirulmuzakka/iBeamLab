# Local validation ? 2026-10-27

Environment: Windows x64, MSVC 19.44, Python 3.11, CPU ONNX Runtime 1.22.1.
All commands use the independent `cpp/CMakeLists.txt`; no legacy library is linked.

| Configuration | Result |
| --- | --- |
| Static, ONNX + SIMNRA compiled, Python/examples/benchmark enabled | 5 test targets passed; licensed SIMNRA reference test skipped |
| Shared, same enabled modules | 5 test targets passed; licensed SIMNRA reference test skipped |
| Static, ONNX/SIMNRA/Python disabled | 4 test targets passed |

Tests cover core validation/parameter updates; rebin conservation/coverage/alias
rejection; pileup direct-convolution parity and inversion; deterministic generation;
record/discard/stop/cancellation policies; sharded storage round trips, incomplete
state and corruption; package validation; ONNX tensor schemas; forward/inverse
single/batch predictions; exposure, calibration and pileup corrections; and
independent installed C++ consumers. Python tests additionally cover ndarray
output buffers, callbacks, package deployment and custom PyTorch export parity
at two batch sizes.

A default Python wheel was built in an isolated build environment, installed
into a separate target directory, and imported successfully with Python isolated
mode. The trained forward example generated 200 records, split train/validation/
test, selected its best validation checkpoint, exported the graph, and verified
physical inference parity. The C++ generation example wrote 100 records.

The 5,000-channel processing benchmark ran 100 rebin+pileup iterations in about
65 ms on this host with zero counted heap allocations after warmup. This is a
local measurement, not a cross-platform speed guarantee. Inverse pileup remains
a quadratic recurrence and was not included in that timing.

Not executed here: licensed SIMNRA numerical verification (no configured trusted
fixture), Linux builds, Linux sanitizer execution, or remote CI runs. The test
and CI entry points are implemented. Fitting remains intentionally unimplemented.
iBeamLab format 1 does not imply compatibility with legacy datasets/ZIP packages.

## Follow-up: direct XNRA verification

The previous fixture prerequisite was unnecessary: `xnra/` already contains both
setup and stored simulation data. SIMNRA 7.04 is registered on this host. The new
`simnra-reference` test reads those files directly, without changing them, and
uses pywin32 only in the test to establish an independent unmodified-project COM
baseline. See `SIMNRA_VALIDATION.md` for the actual numerical results. This
follow-up supersedes the earlier "no configured trusted fixture" skip statement.

## Latest recheck

All five SIMNRA backend cases now match fresh independent calculations. The
NRA failure was traced to XML/COM composition round-trip precision and fixed
in the test input loader, without changing output tolerances or ignoring user
parameter changes. Forced target replacement and a 10% thickness variation pass
against independent calculations for all five cases. One RBS no-pileup saved
spectrum still differs from SIMNRA's own untouched recalculation; the integration
test continues to fail that comparison. `SIMNRA_VALIDATION.md` contains the
current, authoritative per-reference results.
