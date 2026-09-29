# Python quick start

Install a released wheel with `pip install ibeamlab`, or build the current
checkout with `pip install .`. SIMNRA is optional: importing iBeamLab, using the
dummy backend, reading datasets, and running packaged models do not start it.

```python
import ibeamlab as ibl

sample = ibl.Sample([
    ibl.Layer(500_000, {"Li": 0.25, "Ni": 0.20, "Mn": 0.025,
                        "Co": 0.025, "O": 0.50})
])
experiment = ibl.Experiment(sample, [
    ibl.Detector("RBS", ibl.Beam("H", energy=2974), resolution=20)
])

result = ibl.DummySimulator(channels=1024).simulate(experiment)
print(result["RBS"].counts)
```

## Parameter studies

```python
study = ibl.GenerationStudy(experiment, [
    ibl.vary.concentration("Li", layer=0, bounds=(0.1, 0.4)),
    ibl.vary.concentration("Ni", layer=0, bounds=(0.1, 0.4)),
])

summary = study.generate(
    "datasets/example", simulator=ibl.DummySimulator(), samples=1000, seed=42
)
dataset = ibl.open_dataset(summary.path)
print(dataset.parameters.shape, dataset.spectra("RBS").shape)
```

Layer thickness uses `1e15 atoms/cm2`. Beam energy, beam spread, and detector
resolution use keV. Acquisition times use seconds. Concentrations and isotope
fractions are unitless and must sum to one within `1e-6`.

SIMNRA studies pass the same method definitions to the study and backend:

```python
method = ibl.SimnraMethod("RBS", "reference.xnra")
study = ibl.GenerationStudy(experiment, parameters, methods=[method])
with ibl.SimnraSimulator([method], workers=4) as simulator:
    study.generate("datasets/simnra", simulator=simulator, samples=10_000)
```

Direct pybind11 access is intentionally outside the stable API. Advanced users
can explicitly import `ibeamlab.native`.

## TOML-driven multilayer generation

The multilayer example reads every scientific and execution setting from a
versioned TOML file:

```console
python examples/generate_multilayer.py examples/multilayer-generation.toml --validate-only
python examples/generate_multilayer.py examples/multilayer-generation.toml
```

The default example generates 20,000 regional-weight mixtures for each layer
count from 1 through 10. The one-layer case additionally contains 2,000 pure
samples for each of nine elements, giving 38,000 one-layer samples and 218,000
samples overall.

Regional intervals guide raw element weights. Inactive elements are assigned
zero, active weights are drawn from their configured interval, and the complete
layer is normalized to one. The final concentration can therefore occupy a
different region than its proposed raw weight; both proposed and realized
counts are written to `sampling-summary.toml`.

Thickness sampling mirrors the legacy ibamlkit policy: a per-sample total
envelope is drawn, divided into exponentially increasing layer envelopes, and
each actual thickness is drawn independently between zero and its layer
envelope.

Each run writes:

```text
output/
├── generation.toml
├── generation.resolved.toml
├── generation-summary.toml
├── layers_01/
│   ├── dataset.toml
│   ├── sampling-components.csv
│   └── sampling-summary.toml
└── ...
```

`generation.resolved.toml` records software versions, absolute SIMNRA reference
paths, SHA-256 checksums, and the deterministic seed-stream convention. Existing
output directories are rejected to prevent accidental overwrites.
