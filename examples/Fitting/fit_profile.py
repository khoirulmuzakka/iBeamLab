"""Small synthetic profile fit; no SIMNRA or model package required."""
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
    profile = ibl.ProfileModel((ibl.ProfileLayer(
        ibl.Layer(1000, {"Si": .5, "O": .5}),
        (ibl.polynomial(1, -1), ibl.polynomial(0, 1)), sublayers=12),))
    setup = ibl.ExperimentalSetup([ibl.Measurement("RBS", ibl.Beam("H", 2000))])
    simulator = ibl.DummySimulator(channels=64)
    observed = simulator.simulate(ibl.Experiment(profile.discretize(), setup))
    data = ibl.MeasuredData(setup, [ibl.MeasuredSpectrum("RBS", observed["RBS"].counts)])
    problem = ibl.FitProblem(data, profile.reference_sample(), [], simulator, profile=profile,
                profile_parameters=[ibl.ProfileParameter(0, 0, 1, (-1, 0), "silicon_slope")])
    result = ibl.fit(problem, max_evaluations=100, population_size=8)
    print("Synthetic workflow check; DummySimulator is not a physical simulator.")
    print("Cost:", result.cost, "Parameters:", result.parameters)


if __name__ == "__main__":
    main()
