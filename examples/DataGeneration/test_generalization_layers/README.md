# RBS generalization across layer counts

Run from the repository root with the DL environment:

```powershell
python examples/DataGeneration/test_generalization_layers/generate.py
```

Defaults generate 10,000 samples each for 10, 13, 16 and 20 layers using 64
persistent SIMNRA workers, MTA and the calculation mode from the base TOML.
The base configuration is `../multilayer-generation.toml`. The reference is
`xnra/Ref_RBS_LiCOFNaAlSiPSTiMnFeCoNiCuH.xnra`, with pileup enabled and verified.
SIMNRA applies pileup; Poisson noise is drawn afterward without applying pileup again.

The existing compound/shared-random/layerwise-random mixture and concentration
rules are retained. Each system uses the base maximum total thickness envelope
(currently 700,000 in units of 1e15 atoms/cm2), divided across its layers by the
existing exponential-envelope sampler. No additional pure-element samples are generated.

For each sample, independently sample uniform particlesSr within 10% of the base,
linear calibration within 5% of the base, and calibration offset in [-10,10] keV.
These are physical dataset parameter columns, alongside thicknesses and fractions.
Random streams for composition, setup and noise are separate and reproducible.

Each `layers_NN` folder contains:

- `expected/`: native sharded dataset of SIMNRA expected counts, including pileup.
- `noisy/`: matching native dataset with independent Poisson observations.
- `generation.resolved.json`: reference checksum, setup bounds and resolved settings.
- `sampling-summary.json` and a copy of the base TOML.

Read either dataset with `ibeamlab.Dataset(path)`. Row identities and physical
parameters match between expected and noisy data. `generalization-summary.json`
reports completion. Interrupted datasets retain the native incomplete marker;
existing case folders are refused instead of overwritten. Use a new `--output`
directory for another run. `--validate-only` checks configuration and references
without starting SIMNRA. `--layers`, `--samples`, `--workers`, `--seed` and
`--extension-dir` can be overridden.

After completion, validate paired rows, setup bounds and aggregate Poisson statistics:

```powershell
python examples/DataGeneration/test_generalization_layers/verify.py
```

The results are saved to `verification.json`. This streams stored spectra and
does not run SIMNRA again.

## Surrogate comparison

Open `compare_surrogate.ipynb` in a fresh kernel. Select the exported `.ibm` file
and an ONNX-enabled native extension. The notebook defaults to the existing
isolated ONNX build when available; the simulation-only build cannot run inference.
It evaluates all samples by default, with a configurable bounded batch size and
optional preview limit. Each sample's stored calibration and exposure are used.
Only an in-memory copy of the package's rebin grid/coverage policy is adjusted
to cover the measured spectra; its weights, transforms and pileup time are preserved.

The notebook saves per-sample `sum((data-prediction)^2/(1+data))` scores, noise-baseline scores,
model discrepancy from noiseless SIMNRA, distribution plots and two 28-panel
figures (overlays and residuals). Columns represent layer counts and rows select
seven score quantiles, including the worst prediction. Outputs go into a new
timestamped `comparisons/` folder. No fitting is performed; normalized chi-square
divides by the number of evaluated bins, without a fitted-parameter correction.
