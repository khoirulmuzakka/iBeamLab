"""Generate a 50 x 50 experimental-style RBS/NRA scan with SIMNRA.

Each scan position is represented by ten geometrically increasing layers with
a total areal thickness of 500,000 (1e15 atoms/cm2). Output files use the form
``SampleName_Xnm_Theta1_Ynm_Theta2_METHOD.dat``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ibeamlab as ibl
from generate import METHODS, SIMNRA_WORKERS, make_experiment
from spatial_sample import (
    ELEMENTS,
    MAX_DEPTH,
    SAMPLE_SIZE_MM,
    scan_grid,
    sublayer_compositions,
    sublayer_thicknesses,
)
THETA1 = 150
THETA2 = 50
SCAN_POINTS = 50
BATCH_SIZE = SIMNRA_WORKERS


def sample_at(x_mm: float, y_mm: float, thicknesses: np.ndarray) -> ibl.Sample:
    """Materialize the ten-layer sample at one lateral scan position."""
    concentrations = sublayer_compositions(x_mm, y_mm, thicknesses)
    return ibl.Sample([
        ibl.Layer(float(thickness), dict(zip(ELEMENTS, row.tolist(), strict=True)))
        for thickness, row in zip(thicknesses, concentrations, strict=True)
    ])


def experiment_at(x_mm: float, y_mm: float, thicknesses: np.ndarray,
                  detectors: tuple[ibl.Detector, ...]) -> ibl.Experiment:
    return ibl.Experiment(sample_at(x_mm, y_mm, thicknesses), detectors)


def coordinate_nm(value_mm: float) -> int:
    """Convert a grid coordinate to the nearest integer nanometre."""
    return int(round(value_mm * 1_000_000.0))


def write_dat(path: Path, counts: np.ndarray, real_time: float, live_time: float) -> None:
    """Write one spectrum in the experimental text format used in data/."""
    lines = [
        f"Real time: {round(real_time * 1_000_000)} us\n",
        f"Live time: {round(live_time * 1_000_000)} us\n",
        "channel counts\n",
    ]
    lines.extend(f"{channel}\t{int(count)}\n" for channel, count in enumerate(counts))
    path.write_text("".join(lines), encoding="utf-8", newline="\n")


def output_name(sample_name: str, x_mm: float, y_mm: float, method: str) -> str:
    return (
        f"{sample_name}_{coordinate_nm(x_mm)}_{THETA1}_"
        f"{coordinate_nm(y_mm)}_{THETA2}_{method}.dat"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-name", default="SimulatedNMC")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--no-poisson",
        action="store_true",
        help="round expected SIMNRA counts instead of applying Poisson counting noise",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.sample_name or any(char in args.sample_name for char in '<>:"/\\|?*'):
        raise SystemExit("sample name is empty or contains a Windows filename character")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = args.output or ROOT / "examples" / "datasets" / f"spatial_scan_{run_id}"
    output.mkdir(parents=True, exist_ok=False)

    base_experiment = make_experiment()
    detectors = {detector.label: detector for detector in base_experiment.detectors}
    thicknesses = sublayer_thicknesses()
    if not np.isclose(thicknesses.sum(), MAX_DEPTH):
        raise RuntimeError("sublayer thicknesses do not sum to the requested depth")

    x_grid, y_grid = scan_grid(SCAN_POINTS)
    positions = list(zip(x_grid.ravel(), y_grid.ravel(), strict=True))
    rng = np.random.default_rng(args.seed)
    with ibl.SimnraSimulator(METHODS, workers=SIMNRA_WORKERS) as simulator:
        with tqdm(total=len(positions), unit="point", desc="SIMNRA spatial scan") as bar:
            for start in range(0, len(positions), BATCH_SIZE):
                batch_positions = positions[start : start + BATCH_SIZE]
                experiments = [
                    experiment_at(float(x), float(y), thicknesses, base_experiment.detectors)
                    for x, y in batch_positions
                ]
                results = simulator.simulate_many(experiments)
                for (x, y), result in zip(batch_positions, results, strict=True):
                    if result.failed:
                        failure = result.failure
                        raise RuntimeError(
                            f"SIMNRA failed at ({x:g}, {y:g}) for {failure.method_label}: "
                            f"{failure.message}"
                        )
                    for spectrum in result.spectra:
                        expected = np.clip(np.asarray(spectrum.counts, dtype=float), 0.0, None)
                        counts = np.rint(expected) if args.no_poisson else rng.poisson(expected)
                        detector = detectors[spectrum.label]
                        path = output / output_name(
                            args.sample_name, float(x), float(y), spectrum.label
                        )
                        write_dat(path, counts, detector.real_time, detector.live_time)
                bar.update(len(batch_positions))

    print(f"Finished {len(positions)} scan points / {len(positions) * 2} files: {output}")
    print("Layer thicknesses:", ", ".join(f"{value:.6g}" for value in thicknesses))


if __name__ == "__main__":
    main()
