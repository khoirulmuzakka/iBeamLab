"""Fit a measured text spectrum using an exported forward-surrogate package.

Run from the repository root after compiling the Python extension::

    python examples/Fitting/fit_spectrum.py data.txt model.ibm --extension-dir build/local/Release/bin

The example starts from the package's explicit reference setup and sample
assumption. Adjust those descriptions and bounds for the actual experiment.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    """Load measured channels, fit the first layer and report detector costs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spectrum", type=Path)
    parser.add_argument("package", type=Path)
    parser.add_argument("--extension-dir", type=Path)
    parser.add_argument("--sample-json", type=Path, help='Initial sample: {"layers": [{"thickness": 100, "species": {"Si": 1}}]}')
    parser.add_argument("--measurement", default="RBS")
    parser.add_argument("--minimum", type=float, default=1)
    parser.add_argument("--maximum", type=float, default=1000)
    parser.add_argument("--evaluations", type=int, default=10000)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args.extension_dir:
        sys.path.insert(0, str(args.extension_dir.resolve()))
    import ibeamlab as ibl
    model = ibl.load_model(args.package)
    metadata = model.metadata.model
    schema = metadata.input_preparation
    if args.sample_json:
        initial_sample = ibl.Sample([ibl.Layer(**layer) for layer in
                                    json.loads(args.sample_json.read_text(encoding="utf-8"))["layers"]])
    elif isinstance(schema, ibl.ParameterInputSchema):
        initial_sample = schema.reference_sample
    else:
        parser.error("layer-input models require --sample-json with an explicit initial sample")
    if len(metadata.reference_setup.measurements) != 1:
        parser.error("this example accepts one observation; provide every detector in MeasuredData for a joint fit")
    data = ibl.MeasuredData(metadata.reference_setup, [ibl.read_spectrum(
        args.spectrum, measurement=args.measurement, format="channel_counts")])
    problem = ibl.FitProblem(data, initial_sample,
        [ibl.vary.layer_thickness(layer=0, bounds=(args.minimum, args.maximum))], model)
    result = ibl.fit(problem, max_evaluations=args.evaluations,
        progress=lambda p: print(f"Evaluations: {p.evaluations}, chi2: {p.best_cost:.6g}") or False)
    print(result.status, result.parameters, "chi2:", result.cost)
    for contribution in result.contributions:
        print(contribution.measurement, contribution.cost, "channels:", contribution.channels)


if __name__ == "__main__":
    main()
