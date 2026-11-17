# RBS generalization across layer counts

`generate.py` defaults to 1,000 samples each for **1-8, 10, 13, 16 and 20
layers**, stored in `generation_training_rules_1000/`. Independent uniform setup
ranges are particlesSr **[1e10, 1e12]**, linear calibration **[1.0, 3.0]
keV/channel**, and calibration offset **[-10, 10] keV**.

The training compound/shared-random/layerwise-random composition mixture and
concentration rules are retained. The training thickness sampler uses a shared
maximum layer count of 20 and an envelope range of 50,000-500,000. The total upper
envelope is `50000 + (layers - 1) / 19 * 450000`; exponential layer allocation
(growth ratio 1.6) and uniform draws within each layer envelope are unchanged. No extra
pure-element samples are generated. SIMNRA includes pileup; independent Poisson
noise is drawn afterward. Composition, setup and noise use separate reproducible
random streams.

Run from the repository root with the Python environment matching the extension:

```powershell
python3.11 examples/DataGeneration/test_generalization_layers/generate.py --extension-dir build/cuda/Release/bin
python3.11 examples/DataGeneration/test_generalization_layers/verify.py --extension-dir build/cuda/Release/bin
```

The base configuration is `../multilayer-generation.toml`. It supplies the
composition/thickness method, MTA setting, calculation mode, batch size and shard
size. Generation uses 64 persistent SIMNRA workers by default and the pileup-enabled
reference `xnra/Ref_RBS_LiCOFNaAlSiPSTiMnFeCoNiCuH.xnra`.

Use a new `--output` path for another run; existing case folders are refused.
`--layers`, `--samples`, `--workers`, `--seed`, `--particles-sr-range`,
`--calibration-linear-range`, `--calibration-offset-range` and
`--thickness-envelope-range` override defaults. The thickness upper envelope
is interpolated using the largest requested layer count, as in training.
`--validate-only` checks configuration and references without starting SIMNRA.
For example, the current default setup ranges can be given explicitly as:

```powershell
python3.11 examples/DataGeneration/test_generalization_layers/generate.py --extension-dir build/cuda/Release/bin --output NEW_OUTPUT --layers 1 2 3 4 5 6 7 8 10 13 16 20 --samples 1000 --particles-sr-range 1e10 1e12 --calibration-linear-range 1 3 --calibration-offset-range -10 10
```

Each generation contains `generalization-summary.json`, `verification.json`, and
one `layers_NN/` folder per system. Each case contains:

- `expected/`: native sharded SIMNRA expected counts, including pileup.
- `noisy/`: paired spectra with independent Poisson observations.
- `generation.resolved.json`: reference checksum, setup bounds and resolved settings.
- `sampling-summary.json` and a copy of the base TOML.

Read datasets with `ibeamlab.Dataset(path)`. The verifier streams paired rows,
checks identity, setup bounds, counts and aggregate Poisson statistics without
running SIMNRA again. Pass `--output` to verify another generation.

## Surrogate comparison

Open `compare_surrogate.ipynb` in a fresh kernel with an ONNX-enabled extension.
Set `GENERATION_PATH` to choose `generation_training_rules_1000/` (default),
another completed generation.
Layer counts, sample counts, setup bounds and dataset fingerprints are read from
the selected run. 

The notebook evaluates all samples by default in bounded batches through native
inference, using stored samples, exposure and calibration. Geometry is ignored
by using package geometry. An in-memory package copy adjusts rebin capacity and
coverage for recorded channels; weights, transforms and pileup time are preserved.

Results go into a timestamped `comparisons/` folder inside the selected generation.
Outputs include per-spectrum scores, summary statistics, distribution curves,
histograms with means and approximate ideal reduced chi-square silhouettes, and
spectrum overlays/residuals for seven score quantiles per layer count. Histograms
use focused x-axis limits with hidden-tail counts. Overlays and residuals are split
into groups of up to four layer systems per figure.

Scores use `sum((data-prediction)^2/(1+data))` on the expected-count ROI. Normalized
scores divide by valid-channel count, with no fitted-parameter correction. The
noise baseline uses the same ROI and denominator. The default valid-channel
threshold remains expected counts >=1.
