# iBeamLab

iBeamLab is a Python-first toolkit for ion-beam simulation, generated datasets,
spectrum processing, and packaged ONNX inference. Its compiled C++ engine is an
implementation detail; normal Python use requires no knowledge of C++.

## Installation

Install a compatible wheel with `pip install ibeamlab`. To build this checkout,
install CMake 3.24+ and a C++20 compiler, then run `pip install .`.

SIMNRA is optional and Windows-only. The package can be imported, tested with
the dummy simulator, used for dataset access, and used for ONNX inference
without starting SIMNRA.

## Five-minute example

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

Generate and read a dataset:

```python
study = ibl.GenerationStudy(experiment, [
    ibl.vary.concentration("Li", layer=0, bounds=(0.1, 0.4)),
    ibl.vary.concentration("Ni", layer=0, bounds=(0.1, 0.4)),
])
summary = study.generate(
    "datasets/nmc", simulator=ibl.DummySimulator(), samples=1000, seed=42
)
dataset = ibl.open_dataset(summary.path)
print(dataset.parameters.shape, dataset.spectra("RBS").shape)
```

See the [Python quick start](docs/getting-started.md),
[dataset format](docs/dataset-format.md), and [model packages](docs/model-package.md).

## Reproducible multilayer generation

Multilayer collections are configured entirely through TOML:

```console
python examples/generate_multilayer.py examples/multilayer-generation.toml --validate-only
python examples/generate_multilayer.py examples/multilayer-generation.toml
```

The run stores the original configuration, a resolved configuration containing
absolute paths and reference-file checksums, per-dataset sampling audits, and a
final generation summary. See
[multilayer-generation.toml](examples/multilayer-generation.toml) for the full schema.

## Units

- Beam energy, spread, and detector resolution: keV
- Layer thickness: `1e15 atoms/cm2`
- Acquisition time: seconds
- Concentrations and isotope fractions: unitless

The native C++ API remains available to CMake consumers. Python users needing
unstable low-level access can explicitly import `ibeamlab.native`; see
[native architecture](docs/native.md).
