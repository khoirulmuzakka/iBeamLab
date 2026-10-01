"""Generate a 100 x 100 RBS raster scan of the 4 x 5 cell array.

Each scan position is represented by ten geometrically increasing layers with
a total areal thickness of 100,000 (1e15 atoms/cm2). Output files use the form
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


# Standalone spatial-composition model: depth is total areal density in units
# of 1e15 atoms/cm^2. All model constants and helpers live in this file.
MAX_DEPTH = 100_000.0
SUBLAYER_COUNT = 10
SUBLAYER_GROWTH_RATIO = 1.6

def sublayer_thicknesses(
    count: int = SUBLAYER_COUNT,
    total: float = MAX_DEPTH,
    growth_ratio: float = SUBLAYER_GROWTH_RATIO,
) -> np.ndarray:
    """Return geometrically increasing layer thicknesses summing to ``total``."""
    if count < 1 or total <= 0.0 or growth_ratio <= 1.0:
        raise ValueError("count and total must be positive and growth_ratio must exceed one")
    thicknesses = growth_ratio ** np.arange(count, dtype=float)
    thicknesses *= total / thicknesses.sum()
    # Make the stated total exact despite floating-point summation.
    thicknesses[-1] += total - float(thicknesses.sum())
    return thicknesses

# Four-row / five-column IBA array. The first row is the shared graphite
# anode; the remaining rows are the three NMC cathode chemistries.
ARRAY_ELEMENTS = ("Li", "C", "Ni", "Mn", "Co", "O", "Al")
CELL_DIAMETER_MM = 10.0
CELL_GAP_MM = 1.0
HOLDER_MARGIN_MM = 1.0
ARRAY_COLUMNS = 5
ARRAY_ROWS = 4
CELL_PITCH_MM = CELL_DIAMETER_MM + CELL_GAP_MM
ARRAY_WIDTH_MM = ARRAY_COLUMNS * CELL_DIAMETER_MM + (ARRAY_COLUMNS - 1) * CELL_GAP_MM
ARRAY_HEIGHT_MM = ARRAY_ROWS * CELL_DIAMETER_MM + (ARRAY_ROWS - 1) * CELL_GAP_MM
HOLDER_WIDTH_MM = ARRAY_WIDTH_MM + 2 * HOLDER_MARGIN_MM
HOLDER_HEIGHT_MM = ARRAY_HEIGHT_MM + 2 * HOLDER_MARGIN_MM
ARRAY_SOC_VALUES = (0.0, 0.25, 0.50, 0.75, 1.0)
ARRAY_NMC_COMPOSITIONS = {
    "NMC111": (1 / 3, 1 / 3, 1 / 3),
    "NMC622": (0.6, 0.2, 0.2),
    "NMC811": (0.8, 0.1, 0.1),
}
ARRAY_KAPPA = 0.5
ARRAY_RADIAL_WEIGHT = 1.0
ARRAY_DEPTH_WEIGHT = 2.0


def array_cell_layout() -> list[dict[str, object]]:
    """Describe the 20 circular cells and their centers in holder coordinates."""
    rows = (
        ("Graphite anode", "anode", "Graphite"),
        ("NMC111 cathode", "cathode", "NMC111"),
        ("NMC622 cathode", "cathode", "NMC622"),
        ("NMC811 cathode", "cathode", "NMC811"),
    )
    records = []
    for row, (row_name, electrode, chemistry) in enumerate(rows):
        center_y = (1.5 - row) * CELL_PITCH_MM
        for column, soc in enumerate(ARRAY_SOC_VALUES):
            center_x = (column - 2) * CELL_PITCH_MM
            records.append({
                "cell_id": row * ARRAY_COLUMNS + column,
                "row": row,
                "column": column,
                "row_name": row_name,
                "electrode": electrode,
                "chemistry": chemistry,
                "soc": soc,
                "center_x_mm": center_x,
                "center_y_mm": center_y,
            })
    return records


_ARRAY_CELLS = array_cell_layout()
_ARRAY_PROFILE_PARAMETERS: dict[int, tuple[float, float, float, float, float]] = {}


def _array_profile_parameters(cell: dict[str, object]) -> tuple[float, float, float, float, float]:
    """Return mean fraction, bounded mode amplitudes, and allowed Li fraction."""
    eta = float(cell["soc"])
    envelope = 4.0 * eta * (1.0 - eta)
    alpha_r = ARRAY_KAPPA * envelope * ARRAY_RADIAL_WEIGHT / (ARRAY_RADIAL_WEIGHT + ARRAY_DEPTH_WEIGHT)
    alpha_z = ARRAY_KAPPA * envelope * ARRAY_DEPTH_WEIGHT / (ARRAY_RADIAL_WEIGHT + ARRAY_DEPTH_WEIGHT)

    if cell["electrode"] == "anode":
        lower_l, upper_l, host_atoms = 0.0, 1.0, 6.0
        mean_l = lower_l + eta * (upper_l - lower_l)
    else:
        lower_l, upper_l, host_atoms = 0.3, 1.0, 3.0
        mean_l = upper_l - eta * (upper_l - lower_l)
    mean_fraction = mean_l / (mean_l + host_atoms)
    lower_fraction = lower_l / (lower_l + host_atoms)
    upper_fraction = upper_l / (upper_l + host_atoms)

    # Bound the factorized profile over all possible mode values [-1, 1],
    # preserving its mean while keeping local stoichiometry in range.
    corners = np.array([-1.0, 1.0])

    def valid(scale: float) -> bool:
        values = mean_fraction * (1.0 + scale * alpha_r * corners[:, None]) * (
            1.0 + scale * alpha_z * corners[None, :]
        )
        return values.min() >= lower_fraction and values.max() <= upper_fraction

    scale = 1.0
    if not valid(scale):
        low, high = 0.0, 1.0
        for _ in range(60):
            midpoint = 0.5 * (low + high)
            if valid(midpoint):
                low = midpoint
            else:
                high = midpoint
        scale = low
    return mean_fraction, alpha_r * scale, alpha_z * scale, lower_l, upper_l


for _cell in _ARRAY_CELLS:
    _ARRAY_PROFILE_PARAMETERS[int(_cell["cell_id"])] = _array_profile_parameters(_cell)


def array_scan_grid(points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Return an edge-inclusive raster over the complete rectangular holder."""
    if points < 2:
        raise ValueError("points must be at least 2")
    x_axis = np.linspace(-HOLDER_WIDTH_MM / 2, HOLDER_WIDTH_MM / 2, points)
    y_axis = np.linspace(-HOLDER_HEIGHT_MM / 2, HOLDER_HEIGHT_MM / 2, points)
    return np.meshgrid(x_axis, y_axis, indexing="xy")


def _array_cell_ids(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ids = np.full(x.shape, -1, dtype=np.int16)
    for cell in _ARRAY_CELLS:
        dx = x - float(cell["center_x_mm"])
        dy = y - float(cell["center_y_mm"])
        inside = dx * dx + dy * dy <= (CELL_DIAMETER_MM / 2) ** 2
        ids[inside] = int(cell["cell_id"])
    return ids


def array_composition(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return 4x5-array atomic fractions at holder coordinate (x,y,z).

    The aluminum holder fills every point between cell disks. Inside a disk,
    Li follows a bounded separable radial/depth profile at that cell's SOC;
    the host-element fractions use the row's fixed chemistry. Aluminum is
    present only in the holder gaps, not in the electrode cells.
    """
    x, y, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
    if np.any((z < 0.0) | (z > MAX_DEPTH)):
        raise ValueError(f"z must lie between 0 and {MAX_DEPTH:g}")
    if np.any((x < -HOLDER_WIDTH_MM / 2) | (x > HOLDER_WIDTH_MM / 2) |
              (y < -HOLDER_HEIGHT_MM / 2) | (y > HOLDER_HEIGHT_MM / 2)):
        raise ValueError("x,y coordinates must lie inside the aluminum holder")

    cell_ids = _array_cell_ids(x, y)
    result = np.zeros(x.shape + (len(ARRAY_ELEMENTS),), dtype=float)
    result[..., ARRAY_ELEMENTS.index("Al")] = 1.0  # Holder background.
    li_index = ARRAY_ELEMENTS.index("Li")
    graphite_index = ARRAY_ELEMENTS.index("C")
    ni_index = ARRAY_ELEMENTS.index("Ni")
    mn_index = ARRAY_ELEMENTS.index("Mn")
    co_index = ARRAY_ELEMENTS.index("Co")
    oxygen_index = ARRAY_ELEMENTS.index("O")

    for cell in _ARRAY_CELLS:
        cell_id = int(cell["cell_id"])
        selected = cell_ids == cell_id
        if not np.any(selected):
            continue
        dx = x - float(cell["center_x_mm"])
        dy = y - float(cell["center_y_mm"])
        radius = np.hypot(dx, dy)
        f_r = 1.0 - 2.0 * (radius / (CELL_DIAMETER_MM / 2)) ** 2
        f_z = 1.0 - 2.0 * z / MAX_DEPTH
        mean_fraction, alpha_r, alpha_z, _, _ = _ARRAY_PROFILE_PARAMETERS[cell_id]
        n_li = mean_fraction * (1.0 + alpha_r * f_r) * (1.0 + alpha_z * f_z)
        non_li = 1.0 - n_li
        result[..., li_index][selected] = n_li[selected]
        result[..., ARRAY_ELEMENTS.index("Al")][selected] = 0.0

        if cell["electrode"] == "anode":
            result[..., graphite_index][selected] = non_li[selected]
            continue

        a, b, c = ARRAY_NMC_COMPOSITIONS[str(cell["chemistry"])]
        result[..., ni_index][selected] = ((a / 3.0) * non_li)[selected]
        result[..., mn_index][selected] = ((b / 3.0) * non_li)[selected]
        result[..., co_index][selected] = ((c / 3.0) * non_li)[selected]
        result[..., oxygen_index][selected] = ((2.0 / 3.0) * non_li)[selected]
    return result


def array_sublayer_compositions(
    x: float,
    y: float,
    thicknesses: np.ndarray | None = None,
) -> np.ndarray:
    """Return compositions at the areal-depth midpoint of each SIMNRA layer."""
    values = sublayer_thicknesses() if thicknesses is None else np.asarray(thicknesses, float)
    if values.ndim != 1 or np.any(values <= 0.0):
        raise ValueError("thicknesses must be a one-dimensional positive array")
    edges = np.concatenate(([0.0], np.cumsum(values)))
    if not np.isclose(edges[-1], MAX_DEPTH):
        raise ValueError(f"thicknesses must sum to {MAX_DEPTH:g}")
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    return array_composition(x, y, midpoints)

# Keep this experiment setup local to the spatial scan. This script generates
# measured raster spectra directly and does not use the training-data generator.
SIMNRA_WORKERS = 8
REAL_TIME_MEAN_S = 25.88
ACQUISITION_TIME_RELATIVE_SD = 0.02
DEAD_TIME_FRACTION_MIN = 0.10
DEAD_TIME_FRACTION_MAX = 0.25
METHODS = (
    ibl.SimnraMethod(
        "RBS", ROOT / "xnra/Ref_RBS_LiCOFNaAlSiPSTiMnFeCoNiCuH.xnra"
    ),
)


def sample_acquisition_times(rng: np.random.Generator) -> tuple[float, float]:
    """Sample real time and 10–25% dead time; return real/live seconds."""
    real_time = max(0.1, float(rng.normal(
        REAL_TIME_MEAN_S, REAL_TIME_MEAN_S * ACQUISITION_TIME_RELATIVE_SD
    )))
    dead_time_fraction = rng.uniform(DEAD_TIME_FRACTION_MIN, DEAD_TIME_FRACTION_MAX)
    live_time = real_time * (1.0 - dead_time_fraction)
    return real_time, live_time


def make_detectors(real_time: float, live_time: float) -> tuple[ibl.Detector, ...]:
    """Build RBS detector settings with the selected acquisition times."""
    return (
        ibl.Detector("RBS", ibl.Beam("H", 2974), ibl.LinearCalibration(2.63714),
                     resolution=20, particles_sr=1e12,
                     real_time=real_time, live_time=live_time),
    )


THETA1 = 150
THETA2 = 50
SCAN_POINTS = 100
BATCH_SIZE = SIMNRA_WORKERS


def sample_at(x_mm: float, y_mm: float, thicknesses: np.ndarray) -> ibl.Sample:
    """Materialize the holder/cell composition at one raster position."""
    concentrations = array_sublayer_compositions(x_mm, y_mm, thicknesses)
    return ibl.Sample([
        ibl.Layer(float(thickness), dict(zip(ARRAY_ELEMENTS, row.tolist(), strict=True)))
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
    parser.add_argument("--sample-name", default="NMCcellarray")
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

    thicknesses = sublayer_thicknesses()
    if not np.isclose(thicknesses.sum(), MAX_DEPTH):
        raise RuntimeError("sublayer thicknesses do not sum to the requested depth")

    x_grid, y_grid = array_scan_grid(SCAN_POINTS)
    positions = list(zip(x_grid.ravel(), y_grid.ravel(), strict=True))
    rng = np.random.default_rng(args.seed)
    with ibl.SimnraSimulator(METHODS, workers=SIMNRA_WORKERS) as simulator:
        with tqdm(total=len(positions), unit="point", desc="SIMNRA spatial scan") as bar:
            for start in range(0, len(positions), BATCH_SIZE):
                batch_positions = positions[start : start + BATCH_SIZE]
                batch_detectors = []
                for _ in batch_positions:
                    real_time, live_time = sample_acquisition_times(rng)
                    detector_configs = make_detectors(real_time, live_time)
                    batch_detectors.append({
                        detector.label: detector for detector in detector_configs
                    })
                experiments = [
                    experiment_at(float(x), float(y), thicknesses, tuple(detectors.values()))
                    for (x, y), detectors in zip(batch_positions, batch_detectors, strict=True)
                ]
                results = simulator.simulate_many(experiments)
                for (x, y), detectors, result in zip(
                    batch_positions, batch_detectors, results, strict=True
                ):
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

    print(f"Array: {ARRAY_ROWS} rows x {ARRAY_COLUMNS} columns; "
          f"cell diameter {CELL_DIAMETER_MM:g} mm, gap {CELL_GAP_MM:g} mm")
    print(f"Holder scan area: {HOLDER_WIDTH_MM:g} x {HOLDER_HEIGHT_MM:g} mm")
    print(f"Maximum depth: {MAX_DEPTH:g} (1e15 atoms/cm2)")
    print(f"Finished {len(positions)} scan points / {len(positions) * len(METHODS)} files: {output}")
    print("Layer thicknesses:", ", ".join(f"{value:.6g}" for value in thicknesses))


if __name__ == "__main__":
    main()
