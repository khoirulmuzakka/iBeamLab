"""Generate held-out battery-material spectra and a material-specific AutoNRA setup.

The Test_ prefix denotes exclusion from targeted compound sampling, not a
guaranteed exclusion from the regional random composition distribution.
Run with a Python environment matching the installed iBeamLab native engine.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import tomli_w
from tqdm import tqdm

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import generate_spatial_scan_more as scan
from prepare_surr_batt import fit_elements, material_configuration


MATERIAL = "Test_Na2FePO4F"
CHEMISTRY = scan.Chemistry(
    name="Na2FePO4F", electrode="cathode", mobile_ion="Na",
    host={"Fe": 1, "P": 1, "O": 4, "F": 1}, x_min=1.0, x_max=2.0,
    basis="Ideal one-Na Fe2+/Fe3+ window: Na2FePO4F to NaFePO4F; not a measured practical window",
    source_keys=("na2fepof",),
)
TEST_CHEMISTRIES = {
    "NaFePO4": scan.Chemistry("NaFePO4", "cathode", "Na", {"Fe":1,"P":1,"O":4}, 0., 1.,
        "Ideal NaFePO4 to FePO4 one-Na window; not a capacity-calibrated practical window", ("nafepo4",)),
    "Na2FePO4F": CHEMISTRY,
    "FeF3": scan.Chemistry("FeF3", "cathode", "Li", {"Fe":1,"F":3}, 0., 3.,
        "Ideal FeF3 + 3Li to Fe + 3LiF conversion inventory; phase mixtures represented by average elemental fractions", ("fef3",)),
}
SOURCES = {
    "nafepo4": "https://doi.org/10.1039/C6TA01111J",
    "fef3": "https://doi.org/10.1038/s41563-020-00893-1",
    "na2fepof": "https://doi.org/10.1038/nmat2007",
    "cycling": "https://www.sciencedirect.com/science/article/pii/S0272884212014058",
}
ELEMENTS = (*scan.ARRAY_ELEMENTS, "F")
HOLDER_WIDTH_MM = scan.HOLDER_WIDTH_MM
HOLDER_HEIGHT_MM = scan.CELL_DIAMETER_MM + 2 * scan.HOLDER_MARGIN_MM


def cell_layout(chemistry=CHEMISTRY) -> list[dict]:
    return [dict(cell_id=column, row=0, column=column,
                 row_name=f"{chemistry.name} cathode", electrode="cathode", chemistry=chemistry.name,
                 soc=soc, center_x_mm=(column-2)*scan.CELL_PITCH_MM, center_y_mm=0.0)
            for column, soc in enumerate(scan.ARRAY_SOC_VALUES)]


CELLS = cell_layout()


def profile_parameters(soc: float, chemistry=CHEMISTRY) -> tuple[float, float, float]:
    """Match the existing bounded radial/depth atomic-fraction model."""
    x = float(scan.stoichiometry_from_soc(chemistry, soc))
    mean = x / (x + chemistry.host_atoms)
    envelope = 4*soc*(1-soc)
    weights = scan.ARRAY_RADIAL_WEIGHT + scan.ARRAY_DEPTH_WEIGHT
    ar = scan.ARRAY_KAPPA * envelope * scan.ARRAY_RADIAL_WEIGHT / weights
    az = scan.ARRAY_KAPPA * envelope * scan.ARRAY_DEPTH_WEIGHT / weights
    lower = chemistry.x_min/(chemistry.x_min+chemistry.host_atoms)
    upper = chemistry.x_max/(chemistry.x_max+chemistry.host_atoms)
    corners = np.array([-1., 1.])

    def valid(scale):
        values = mean*(1+scale*ar*corners[:, None])*(1+scale*az*corners[None, :])
        return values.min() >= lower and values.max() <= upper

    scale = 1.0
    if not valid(scale):
        lo, hi = 0., 1.
        for _ in range(60):
            mid = (lo+hi)/2
            if valid(mid):
                lo = mid
            else:
                hi = mid
        scale = lo
    return mean, scale*ar, scale*az


PROFILES = {cell["cell_id"]: profile_parameters(cell["soc"]) for cell in CELLS}


def cell_ids(x, y):
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    ids = np.full(x.shape, -1, dtype=np.int16)
    for cell in CELLS:
        inside = (x-cell["center_x_mm"])**2 + y*y <= (scan.CELL_DIAMETER_MM/2)**2
        ids[inside] = cell["cell_id"]
    return ids


def array_composition(x, y, z, chemistry=CHEMISTRY):
    x, y, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
    if np.any(~np.isfinite(x) | ~np.isfinite(y) | ~np.isfinite(z)):
        raise ValueError("Coordinates must be finite")
    if np.any((abs(x)>HOLDER_WIDTH_MM/2) | (abs(y)>HOLDER_HEIGHT_MM/2) | (z<0) | (z>scan.MAX_DEPTH)):
        raise ValueError("Coordinate outside the holder/depth domain")
    result = np.zeros(x.shape+(len(ELEMENTS),), float)
    result[..., ELEMENTS.index("Al")] = 1.0
    ids = cell_ids(x, y)
    for cell in CELLS:
        selected = ids == cell["cell_id"]
        if not np.any(selected):
            continue
        radius = np.hypot(x-cell["center_x_mm"], y)
        fr = 1-2*(radius/(scan.CELL_DIAMETER_MM/2))**2
        fz = 1-2*z/scan.MAX_DEPTH
        mean, ar, az = profile_parameters(cell["soc"], chemistry)
        ion = mean*(1+ar*fr)*(1+az*fz)
        result[selected] = 0.0
        result[..., ELEMENTS.index(chemistry.mobile_ion)][selected] = ion[selected]
        for e, atoms in chemistry.host.items():
            result[..., ELEMENTS.index(e)][selected] = ((1-ion)*atoms/chemistry.host_atoms)[selected]
    return result


def scan_grid(points=100):
    if points < 5:
        raise ValueError("Use at least five x points")
    y_points = max(2, round(points/5))
    return np.meshgrid(np.linspace(-HOLDER_WIDTH_MM/2, HOLDER_WIDTH_MM/2, points),
                       np.linspace(-HOLDER_HEIGHT_MM/2, HOLDER_HEIGHT_MM/2, y_points), indexing="xy")


def fit_configuration(template: dict, names: list[str], output: Path, chemistry=CHEMISTRY) -> dict:
    """Add the supported F element, then use the existing config-pruning rules."""
    template = copy.deepcopy(template)
    config = template["Config"]
    if "F" not in [s.split(":")[0].strip() for s in config["Layers"]["ElementsPerLayer"][0].split(",")]:
        config["Layers"]["ElementsPerLayer"][0] += ", F:0.1"
        config["Analysis"]["Parameters"].extend([
            ["Erf_1_ F_A", 0.1, 0., 1., "FREE"],
            ["Erf_1_ F_a", 0., -1., 1., "FIXED"],
            ["Erf_1_ F_b", 0., -2., 2., "FIXED"],
            ["Erf_1_ F_c", 1., 0., 1., "FIXED"],
        ])
    return material_configuration(template, chemistry, names, output)


def generate(output: Path, template_path: Path, *, points=100, seed=1, workers=8, no_poisson=False, material="Na2FePO4F"):
    chemistry = TEST_CHEMISTRIES[material]
    material_folder = "Test_" + material
    cells = cell_layout(chemistry)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if workers < 1:
        raise ValueError("Workers must be positive")
    with template_path.open("rb") as stream:
        template = tomllib.load(stream)
    grid_x, grid_y = scan_grid(points)
    coordinates = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    names = [scan.output_name(material_folder+"cellarray", x, y, "RBS").removesuffix("_RBS.dat")
             for x, y in coordinates]
    config = fit_configuration(template, names, output, chemistry)
    for method in config["Config"]["IBA"]["Methods"].values():
        if not Path(method["Reference_file"]).is_file():
            raise FileNotFoundError(method["Reference_file"])
    with (scan.ROOT / "examples/multilayer-generation.toml").open("rb") as stream:
        training = tomllib.load(stream)
    training_elements = {e["symbol"] for e in training["elements"]}
    required = {"Na", *chemistry.host}
    if not required <= training_elements:
        raise ValueError("Test material contains an element outside the training element schema")
    for subset in training["composition_sampling"]["compound_sampling"]["subsets"]:
        if required <= set(subset["elements"]):
            raise ValueError("Material element set is covered by a named training subset")
    t = scan.sublayer_thicknesses()
    edges = np.r_[0, np.cumsum(t)]
    midpoints = (edges[:-1]+edges[1:])/2
    fractions = array_composition(coordinates[:, 0, None], coordinates[:, 1, None], midpoints[None, :], chemistry)
    inventories = np.einsum("l,nle->ne", t, fractions)
    ids = cell_ids(coordinates[:, 0], coordinates[:, 1])
    if points == 100 and any(not np.any(ids == c["cell_id"]) for c in cells):
        raise ValueError("A SOC disk is missing from the raster")
    free = [p[0] for p in config["Config"]["Analysis"]["Parameters"] if p[-1] == "FREE"]
    metadata = dict(schema_version=1, material_folder=material_folder, chemistry=asdict(chemistry),
        elements=fit_elements(chemistry), truth_elements=ELEMENTS, cells=cells,
        scan_columns=grid_x.shape[1], scan_rows=grid_x.shape[0], number_of_spectra=len(names),
        holder_width_mm=HOLDER_WIDTH_MM, holder_height_mm=HOLDER_HEIGHT_MM,
        sources=SOURCES, thicknesses=t.tolist(), free_parameters=free, seed=seed,
        rng="Independent numpy SeedSequence([seed, position_index]) per raster point",
        generator_path=Path(__file__).resolve().as_posix(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_generator_sha256=hashlib.sha256(Path(scan.__file__).read_bytes()).hexdigest(),
        template=template_path.resolve().as_posix(), template_sha256=hashlib.sha256(template_path.read_bytes()).hexdigest(),
        generation="Fresh SIMNRA spectra with experimental-format headers and optional Poisson counting noise",
        generation_status="running", completed_spectra=0,
        generalization="Absent from named compound subsets; regional random training may include similar compositions",
        profile=dict(variable=f"{chemistry.mobile_ion} atomic fraction", kappa=scan.ARRAY_KAPPA,
                     radial_weight=scan.ARRAY_RADIAL_WEIGHT, depth_weight=scan.ARRAY_DEPTH_WEIGHT),
        model_assumptions=[chemistry.basis,
            f"Only {chemistry.mobile_ion} varies independently; host elemental ratios remain fixed",
            "Fixed total areal density; no swelling, SEI, binder or carbon coating"],
        simulation=dict(reference_file=str(scan.METHODS[0].reference_file), workers=workers,
            poisson=not no_poisson, beam_particle="H", beam_energy_kev=2974.,
            resolution_kev=20., particles_sr=1e12, calibration_linear=2.63714),
        anra=(output / "fit" / f"surr_{material_folder}.anra").resolve().as_posix())
    # Start the real simulator before creating output: unavailable native/COM
    # dependencies must not leave a settings-only folder looking complete.
    with scan.ibl.SimnraSimulator(scan.METHODS, workers=workers) as simulator:
        (output / "data").mkdir(parents=True)
        (output / "fit").mkdir()
        (output / "fit" / f"surr_{material_folder}.anra").write_text(tomli_w.dumps(config), encoding="utf-8")
        meta_path = output / "scan_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
        records = []
        with tqdm(total=len(names), desc=material_folder, unit="spectra") as progress:
            for start in range(0, len(names), workers):
                indices = range(start, min(start+workers, len(names)))
                experiments, rngs, times = [], [], []
                for i in indices:
                    rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
                    real, live = scan.sample_acquisition_times(rng)
                    layers = [scan.ibl.Layer(float(width), dict(zip(ELEMENTS, f.tolist(), strict=True)))
                              for width, f in zip(t, fractions[i], strict=True)]
                    experiments.append(scan.ibl.Experiment(scan.ibl.Sample(layers), scan.make_detectors(real, live)))
                    rngs.append(rng)
                    times.append((real, live))
                results = simulator.simulate_many(experiments)
                for i, result, rng, (real, live) in zip(indices, results, rngs, times, strict=True):
                    if result.failed:
                        raise RuntimeError(f"SIMNRA failed for {names[i]}: {result.failure}")
                    expected = np.asarray(result["RBS"].counts, float)
                    if not np.all(np.isfinite(expected)) or np.any(expected < 0):
                        raise ValueError(f"Invalid SIMNRA counts: {names[i]}")
                    counts = np.rint(expected) if no_poisson else rng.poisson(expected)
                    path = output / "data" / f"{names[i]}_RBS.dat"
                    scan.write_dat(path, counts, real, live)
                    cell = cells[int(ids[i])] if ids[i] >= 0 else None
                    ion = inventories[i, ELEMENTS.index(chemistry.mobile_ion)]/scan.MAX_DEPTH
                    soc = float(scan.soc_from_mobile_fraction(chemistry, ion)) if cell else None
                    records.append(dict(sample=names[i], spectrum_file=path.name,
                        spectrum_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        x_mm=float(coordinates[i, 0]), y_mm=float(coordinates[i, 1]),
                        cell_id=int(ids[i]), region="electrode" if cell else "Al_holder",
                        column=cell["column"]+1 if cell else "", nominal_soc_pct=100*cell["soc"] if cell else "",
                        truth_depth_integrated_soc_pct=100*soc if cell else "",
                        real_time_s=real, live_time_s=live, channels=len(counts),
                        **{f"truth_{e}_atomic_fraction":float(v/scan.MAX_DEPTH)
                           for e, v in zip(ELEMENTS, inventories[i], strict=True)}))
                    progress.update(1)
        with (output / "spectrum_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        metadata.update(generation_status="complete", completed_spectra=len(records))
        meta_path.write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    readme = f"""# {material_folder}: generalization dataset

Five {chemistry.name} cathode disks at 0, 25, 50, 75 and 100% SOC, on an Al holder.
Fresh SIMNRA spectra: {len(names)} positions on a {grid_x.shape[1]} x {grid_x.shape[0]} raster.
Data: data/*.dat; AutoNRA setup: fit/surr_{material_folder}.anra.
Ground truth: scan_metadata.json and spectrum_manifest.csv.

{chemistry.basis}
On charging, x decreases from {chemistry.x_max:g} to {chemistry.x_min:g}.
Host formula (excluding mobile {chemistry.mobile_ion}): {chemistry.host}.
Radial/depth heterogeneity and ten geometric sublayers follow the original scan model.
Fit elements: {', '.join(fit_elements(chemistry))}; {len(free)} free parameters.
Only {chemistry.mobile_ion} has free depth-shape coefficients.
Seven logarithmic fitting sublayers, the existing surrogate and ARRDE are retained.
No fit has been run. plot.ipynb discovers Test_ folders and shows pending results.

Test_ denotes absence from named compound sampling, not guaranteed exclusion
from regional random training compositions. Training configuration/model are not modified.
Sources: {', '.join(SOURCES[k] for k in chemistry.source_keys)}
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(f"Generated {len(names)} experimental-format spectra in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", choices=tuple(TEST_CHEMISTRIES), default="Na2FePO4F")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--template", type=Path, default=Path("E:/Test_data/Batt/LRN_ad/ref_lrn.anra"))
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-poisson", action="store_true")
    args = parser.parse_args()
    output = args.output or scan.ROOT / "examples/datasets/surr_batt" / ("Test_"+args.material)
    generate(output, args.template, points=args.points, seed=args.seed,
             workers=args.workers, no_poisson=args.no_poisson, material=args.material)


if __name__ == "__main__":
    main()
