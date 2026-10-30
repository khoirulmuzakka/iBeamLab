"""Stream paired datasets and verify setup bounds, counts and Poisson statistics."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))


def verify(folder, extension):
    sys.path.insert(0, str(extension.resolve()))
    import numpy as np
    from ibeamlab._engine import engine
    native = engine()
    summary = json.loads((folder / "generalization-summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("generation has not completed")
    report = {}
    for layers in summary["layers"]:
        case = folder / f"layers_{layers:02d}"
        resolved = json.loads((case / "generation.resolved.json").read_text())
        expected = native.DatasetReader(case / "expected")
        noisy = native.DatasetReader(case / "noisy")
        assert expected.metadata.accepted == noisy.metadata.accepted == summary["samples_per_case"]
        lower = np.array([p["bounds"][0] for p in resolved["varied_parameters"]])
        upper = np.array([p["bounds"][1] for p in resolved["varied_parameters"]])
        residual_sum = expected_sum = pearson_sum = 0.
        cells = rows = 0
        minimum_channels, maximum_channels = sys.maxsize, 0
        minimum = np.full(3, np.inf)
        maximum = np.full(3, -np.inf)
        for x, y in zip(expected, noisy, strict=True):
            assert x.sample_index == y.sample_index and x.sample_id == y.sample_id
            parameters = np.asarray(x.parameters)
            np.testing.assert_array_equal(parameters, y.parameters)
            setup = parameters[-3:]
            assert np.all(setup >= lower) and np.all(setup <= upper)
            minimum = np.minimum(minimum, setup)
            maximum = np.maximum(maximum, setup)
            a, b = np.asarray(x.result.spectra[0].counts_array), np.asarray(y.result.spectra[0].counts_array)
            assert a.shape == b.shape and np.all(np.isfinite(a)) and np.all(a >= 0)
            assert np.all(b >= 0) and np.all(b == np.floor(b))
            residual = b - a
            expected_sum += a.sum()
            residual_sum += residual.sum()
            mask = a >= 1
            pearson_sum += np.sum(residual[mask] ** 2 / a[mask])
            cells += int(mask.sum())
            rows += 1
            minimum_channels = min(minimum_channels, len(a))
            maximum_channels = max(maximum_channels, len(a))
        assert rows == summary["samples_per_case"] and expected_sum > 0 and cells > 0
        z = float(residual_sum / np.sqrt(expected_sum))
        pearson = float(pearson_sum / cells)
        assert abs(z) < 8 and .95 < pearson < 1.05, (z, pearson)
        report[str(layers)] = dict(samples=rows, channels=[minimum_channels, maximum_channels],
            setup_minimum=minimum.tolist(), setup_maximum=maximum.tolist(),
            poisson_total_residual_z=z, poisson_mean_squared_standardized_residual=pearson)
        print(layers, report[str(layers)], flush=True)
    (folder / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--extension-dir", type=Path, default=ROOT / "build/local/Release/bin")
    args = parser.parse_args()
    verify(args.output, args.extension_dir)
