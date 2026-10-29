# SIMNRA reference validation ? 2026-10-27

Current result: **all five cases match independent fresh SIMNRA calculations**.
Forced rebuilds, repeated worker/session use, and a 10% first-layer thickness change also pass for all five.
**Four saved spectra pass; one saved-spectrum comparison still fails.** CTest therefore still reports a failure.

Environment: installed SIMNRA 7.04 on Windows. Reference tolerance remains
`rtol=1e-5, atol=1e-5`; worker repeatability uses `rtol=1e-10, atol=1e-10`.
The repository XNRA files have not been modified.

| Reference | Fresh baseline max error | Forced rebuild max error | Changed thickness max error | Saved snapshot |
| --- | ---: | ---: | ---: | --- |
| `Ref_NRA_LiCOFNaAlSiPSTiMnFeCoNiCuH_nopu.xnra` | 0 | 1.8189894e-11 | 0 | PASS |
| `Ref_RBS_LiCOFNaAlSiPSTiMnFeCoNiCuH.xnra` | 0 | 7.27595761e-12 | 0 | PASS |
| `Ref_RBS_LiCOFNaAlSiPSTiMnFeCoNiCuH_nopu.xnra` | 0 | 7.27595761e-12 | 0 | FAIL |
| `Ref_all_nra_nopu.xnra` | 2.27373675e-13 | 2.27373675e-13 | 1.77635684e-15 | PASS |
| `Ref_all_rbs_nopu.xnra` | 7.10542736e-15 | 7.10542736e-15 | 4.4408921e-16 | PASS |

All errors in the table are absolute counts.

## NRA discrepancy: identified and fixed in the reference test

The earlier NRA mismatch was caused by comparing different floating-point input
representations. SIMNRA normalizes decimal compositions when loading XNRA.
The original test read the printed XML fractions and wrote those values back.
For the NRA single-layer project, that changed concentrations by at most
`5.551115123125783e-17`, yet reproduced a 0.306531-count spectral difference.

This was independently reproduced through pywin32 without the iBeamLab library:
replaying the effective COM concentrations leaves the spectrum unchanged;
replaying the rounded XML concentrations produces the discrepancy. The test now
uses the effective loaded concentrations and rejects differences from XML beyond
16 floating-point epsilons relative to each concentration. This aligns the
inputs, rather than widening output tolerance or altering the backend's parameter semantics.
The NRA backend result now matches its fresh baseline exactly; its forced-rebuild
maximum error is approximately 1.82e-11 counts.

Reproduce the independent diagnostic:

```console
python cpp/tests/simnra_precision_probe.py xnra/Ref_NRA_LiCOFNaAlSiPSTiMnFeCoNiCuH_nopu.xnra
```

## Remaining saved-spectrum discrepancy

`Ref_RBS_LiCOFNaAlSiPSTiMnFeCoNiCuH_nopu.xnra` contains a saved total spectrum
which differs from an untouched independent SIMNRA recalculation by up to
0.272576675 counts (relative L1 error 2.13e-6). The library matches that fresh
recalculation exactly. This difference is reproducible without the iBeamLab library.
Its underlying cause is not established; the saved file is not silently refreshed
and the check remains failed. Fast calculation did not reproduce the saved curve.

## Runtime behavior checked

- Replacements are appended and fully populated before removing the original layer prefix.
- Forced rebuilding is tested with `reuse_target_topology=false`, so cached updates cannot bypass it.
- Each reference is exercised with two workers and repeated requests.
- A 10% first-layer thickness reduction is independently applied in COM and compared with the backend.
- Live `PUCalculation` is false for every `nopu` reference and true for the other RBS reference.
- Other CTest targets (core, workflows, installed consumer, allocation benchmark, Python API) pass.

An earlier exploratory species-order reversal changed spectra. Its cause remains
unestablished; exact-target rebuild validation does not establish arbitrary
species-order invariance. It remains listed separately in TODO.md.

## Reproduce the full comparison

```console
ctest --test-dir cpp/build -C Release -R simnra-reference --output-on-failure
```

Machine-readable results: `cpp/build/simnra-xnra-results.json`.
Run standalone SIMNRA diagnostics separately from this integration test.

## CI acceptance after confirmation that saved curves may be outdated

Saved-curve comparison is now diagnostic by default. The licensed integration
suite still requires fresh independent-baseline, worker, rebuild and parameter
update checks. Set `IBEAMLAB_CHECK_SAVED_SIMNRA_SPECTRA=ON` for the previous strict
stored-curve behavior. Earlier failure exit codes in this report refer to that
strict audit, not the default CI acceptance policy. Reference files and numerical
comparison tolerances are unchanged.
