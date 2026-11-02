"""Shared-sample fit using two independent synthetic measurement providers."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-dir", type=Path)
    args = parser.parse_args()
    if args.extension_dir:
        sys.path.insert(0, str(args.extension_dir.resolve()))
    import ibeamlab as ibl
    sample = ibl.Sample([ibl.Layer(100, {"Si": 1})])
    measurements = [ibl.Measurement("RBS", ibl.Beam("H", 2000)),
                    ibl.Measurement("ERDA", ibl.Beam("He", 3000))]
    providers = [ibl.DummySimulator(channels=32), ibl.DummySimulator(channels=16)]
    observations = []
    for measurement, provider in zip(measurements, providers):
        result = provider.simulate(ibl.Experiment(sample, ibl.ExperimentalSetup([measurement])))
        observations.append(ibl.MeasuredSpectrum(measurement.label, result[measurement.label].counts))
    predictor = ibl.CompositePredictor({m.label: p for m, p in zip(measurements, providers)})
    data = ibl.MeasuredData(ibl.ExperimentalSetup(measurements), observations)
    problem = ibl.FitProblem(data, sample, [ibl.vary.layer_thickness(layer=0, bounds=(50, 150))], predictor)
    result = ibl.fit(problem, max_evaluations=100, population_size=8)
    print("Synthetic workflow check; replace providers with physical backends/models.")
    print("Cost:", result.cost, "Parameters:", result.parameters)


if __name__ == "__main__":
    main()
