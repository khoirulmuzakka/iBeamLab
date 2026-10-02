# Bare forward spectra and measurement corrections

Export a fixed-setup model trained on spectra generated without pileup with
`model.export(..., bare_spectrum_corrections=True)`. This explicitly declares
reference channel spectra. Generic exports and older packages retain their
existing raw inference behavior.

The package stores the reference setup, `bare_spectrum_corrections`,
`Apply_pileup_on_inference` (default true), and
`pileup_fudge_factor_seconds` (default 0.4e-6 seconds).
AutoNRA's 0.4 setting is multiplied by 1e-6 in its implementation; supply
0.4e-6 here for identical physical units and results.

Native forward prediction restores output scaling, conservatively rebins
reference channel edges to the requested calibration, scales by requested /
reference ParticlesSr, and applies pileup. Channel edges use
E(c) = offset + linear*c + quadratic*c*c. Pileup extends the spectrum to
2*N-1 channels. Rebinning can also change the returned spectrum length.
Output bins outside the reference energy coverage contain zero counts.

Use `ForwardModel(path, apply_pileup_on_inference=False)` or the same keyword
on `load_model` in Python; in C++ set
`InferenceOptions::applyPileupOnInference = false`. Calibration and exposure
corrections still run. With pileup enabled, real time must be positive and live
time nonnegative, supplied on each requested detector.

The beam particle, energy, spread, and detector resolution remain tied to the
training setup. Mismatches warn once per distinct detector warning per loaded
model; predictions still use training values. Unsupported elements warn once
per element and layer and are ignored by the surrogate; missing supported
elements have zero concentration. Fewer layers are zero padded; excess layers
are rejected. These warnings do not imply accuracy for unsupported samples.

Fixed-setup correction packages reject setup parameters as network inputs to
prevent applying exposure or calibration changes twice. Spectrum labels must
match reference detector labels. AutoNRA remains independent reference code.

## Batch threading

For eight concurrent fits on a 128-thread CPU, start with
`ForwardModel(path, threads=16, correction_threads=16)` in Python (also accepted
by `load_model`). In C++, set `InferenceOptions::intraOpThreads = 16`,
`interOpThreads = 1`, and `correctionThreads = 16`.

ONNX inference completes before corrections start. With OpenMP available,
corrections run across candidates with at most min(correctionThreads, batch size)
workers, preserving result order. Each worker owns its temporary buffers;
exceptions are rethrown after workers finish. The default is one correction
thread. Builds without OpenMP use serial corrections. Thread counts are local
to the call; no global OpenMP setting is changed. Benchmark overall throughput
under the actual number of simultaneous fits.
