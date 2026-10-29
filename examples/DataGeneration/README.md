# Multilayer data generation

This example adapts the reference multilayer workflow to the current Python
package and C++ core. It generates a separate dataset for each layer count,
using the same ordered element columns for every layer.

Files:

- `multilayer-generation.toml`: sampling, experiment, output and backend settings.
- `generate_multilayer.py`: command-line entry point.
- `configuration.py`: example-specific configuration validation and orchestration.
- `sampling.py`: regional/compound composition sampling and thickness envelopes.

## Run

Install the package from the repository root with `python -m pip install .`.
Python 3.11+ includes TOML support; on Python 3.10, install `.[examples]`.

Validate settings and required reference files without loading the C++ extension
or starting SIMNRA:

```console
python examples/DataGeneration/generate_multilayer.py --validate-only
```

Run a small synthetic example without SIMNRA:

```console
python examples/DataGeneration/generate_multilayer.py --backend dummy --maximum-layers 2 --mixed-samples 8 --pure-samples 2 --output examples/DataGeneration/datasets/smoke
```

For source-tree development, append `--extension-dir build/local/bin` instead of
installing a wheel. For `compile.bat Release`, use
`--extension-dir build/local/Release/bin`. `--help` lists all options. A custom TOML file can be passed
as the first argument. Relative paths inside TOML resolve against its directory;
the command-line `--output` resolves against the working directory.

With the SIMNRA-enabled extension and a local SIMNRA installation:

```console
python examples/DataGeneration/generate_multilayer.py
```

The supplied TOML uses 13 elements, layer counts 1 through 8, 128 mixed samples
per layer count, and 16 additional pure samples per element for the one-layer
dataset: **1,232 total samples**. These are example-sized budgets. The reference
production budgets were 300,000 mixed samples per layer count and 10,000 pure
samples per element; edit the TOML or use the command-line overrides to scale up.
Existing output directories are refused.

## Sampling and parameter layout

Each layer contributes thickness followed by all configured element fractions
in TOML order. Concentrations are explicitly coupled and normalized before the
rows reach the core. The core does not normalize them again.

- Each composition method has its own TOML section and fraction. The three
  fractions must sum to one. Largest-remainder rounding allocates the exact
  mixed-sample budget; equal remainders follow section order: compound, shared
  random elements, then random elements.
- `composition_sampling.compound`: choose a configured element subset per
  sample and use that set in every layer. Subset counts are proportional to
  subset size. Names identify element sets, not fixed stoichiometry.
- `composition_sampling.shared_random_elements`: uniformly choose an element
  count from **2 through 8**, then choose that many distinct elements uniformly.
  This element set is shared by every layer of that sample.
- `composition_sampling.random_elements`: independently choose an element
  count from **1 through 13** and a distinct element set for **each layer**.
- All three methods draw concentrations independently between layers, using
  the shared per-element trace/minor/major/dominant region probabilities.
  Regions describe raw weights; normalized fractions can move to another
  region. Selected elements have positive fractions; all absent elements retain
  zero-valued columns. Element activation probabilities are not used.
- Pure-element samples are additional and appear only in the one-layer dataset.
  Their thicknesses are stratified over the one-layer envelope.
- Mixed thicknesses use a sampled total envelope divided exponentially among
  layers. Each layer draws uniformly inside its envelope, so actual total
  thickness may be below the configured minimum total envelope.
- Named PCG64 streams depend on the master seed, layer count and sampling
  component. Sampling one layer count does not advance another's streams.

The supplied fractions are **0.5 compound, 0.25 shared random elements, and
0.25 random elements**: 64/32/32 samples for each 128-row mixed budget. Edit
`fraction` inside each method's section to change the split. Random element
count bounds are inclusive and must fit within the configured element list;
only uniform counts and uniform selection without replacement are supported.

Each measurement has its own beam, detector and acquisition settings. Add
entries under `experiment.measurements` and matching `simulation.methods` to
generate multiple spectra per sample. Reference particles must match their
measurement beams. The `nopu` reference has pileup disabled.

The dummy backend uses `simulation.channels` for every measurement; SIMNRA
takes channel counts from each reference. `records_per_shard` controls writer
buffering. `simulation.batch_size` independently limits the number of rows
submitted to the backend at once (default 512, supported range 1-4096). Choose
a batch at least as large as `workers` to keep the SIMNRA pool busy. Results
retain original row order even when calculations finish out of order.

`simulation.fast_calculation=true` selects `CalculateSpectrumFast`, matching
the reference generator. This is a different numerical algorithm, not just a
display optimization: spectra can differ from `CalculateSpectrum`. Set it to
false for the standard algorithm. The standalone simulator API defaults to
standard calculation; the example makes the fast choice explicit. Elemental
spectra are disabled because this workflow stores total spectra only.

Each layer-count system has its own tqdm bar, spectrum rate and ETA. With
multiple measurements, each processed sample advances by its number of spectra.
Initial worker/reference startup is included in the first progress estimates.
Rows and sampling arrays are materialized for one layer-count dataset at a time;
in-flight experiments and results are bounded by batch_size. Ctrl+C drains the
in-flight jobs safely and leaves an incomplete dataset. Choose a new output
directory for a new run; incomplete output is never silently overwritten.

## Output

```text
output/
  generation.toml
  generation.resolved.json
  generation-summary.json
  layers_01/
    manifest.ibd
    shard-0.ibd
    ...
    sampling-components.csv
    sampling-summary.json
  layers_02/
    ...
```

The original TOML is preserved. Resolved JSON records command-line overrides,
absolute paths, reference/configuration hashes, seeds and software versions.
Per-layer sampling artifacts describe the requested rows and their method and
component labels. The CSV also records JSON lists of each layer's selected
elements and element counts. Sampling summaries include per-method budgets,
element-count histograms and element occurrences. Core records retain the
original `sample_index` even when rows fail or
are discarded. `generation-summary.json` reports accepted/failed/discarded
counts and completion. On failure or cancellation, output is preserved with
`complete=false`; incomplete core datasets cannot be opened as completed data.

Read completed data with the new wrapper:

```python
import ibeamlab as ibl

dataset = ibl.open_dataset("examples/DataGeneration/datasets/smoke/layers_02")
print(dataset.metadata.parameter_names)
for batch in dataset.iter_batches(32):
    print(batch[0].parameters, batch[0].result["RBS"].counts.shape)

parameters = dataset.parameters     # successful rows, physical float64
spectra = dataset.spectra("RBS")    # same successful row order
```
