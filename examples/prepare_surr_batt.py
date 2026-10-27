"""Split the existing synthetic battery raster into material-specific fits.

Each material receives a 100-column by 20-row band containing its five SOC
disks and Al holder. Original spectra, coordinate names and counting noise
are retained byte for byte. No simulator or optimizer is run.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import tomli_w

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import generate_spatial_scan_more as scan


MATERIAL_FOLDERS = {
    "NMC111": "NMC111", "LFP": "LFP", "LMO": "LMO",
    "Sodium NFM": "Sodium_NFM", "Graphite": "Graphite",
    "Silicon": "Silicon", "Silicon-carbon": "Silicon_carbon",
    "Hard carbon": "Hard_carbon",
}
PARAMETER = re.compile(r"Erf_1_\s*([A-Z][a-z]?)_([Aabc])")
SAMPLE = re.compile(r"^(.+)_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)$")


def fit_elements(chemistry: scan.Chemistry) -> tuple[str, ...]:
    """Only the material's ion/host elements and the aluminium holder."""
    return (chemistry.mobile_ion, *(e for e in chemistry.host if e != "O"),
            "Al", *(("O",) if "O" in chemistry.host else ()))


def material_configuration(template: dict, chemistry: scan.Chemistry,
                           names: list[str], material_dir: Path) -> dict:
    """Keep the working optimizer/settings; remove inactive element parameters."""
    doc = {"Config": copy.deepcopy(template["Config"])}
    config = doc["Config"]
    layers = config["Layers"]
    if (layers["NumberOfLayers"] != 1 or layers["Parameterization"] != ["Erf-polynomial"]
            or layers["Parameterization_Degree"] != [0]):
        raise ValueError("Expected the working single-layer degree-zero Erf template")
    analysis = config["Analysis"]
    if analysis["Parameter_constraints"]:
        raise ValueError("Template constraints require review before pruning elements")
    elements = fit_elements(chemistry)
    initial = {item.split(":")[0].strip(): item.split(":")[1].strip()
               for item in layers["ElementsPerLayer"][0].split(",")}
    if set(elements) - set(initial):
        raise ValueError(f"Template lacks elements for {chemistry.name}")
    layers["ElementsPerLayer"] = [", ".join(f"{e}:{initial[e]}" for e in elements)]
    parameters = []
    for entry in analysis["Parameters"]:
        p = list(entry)
        match = PARAMETER.fullmatch(p[0])
        if match:
            element, coefficient = match.groups()
            if element not in elements:
                continue
            # Keep all allowed amplitudes independent. Do not impose host ratios
            # or give the fit true SOC. Only the mobile ion has shape freedom.
            if coefficient in "abc" and element != chemistry.mobile_ion:
                p[-1] = "FIXED"
        parameters.append(p)
    expected = {(e, c) for e in elements for c in "Aabc"}
    actual = {m.groups() for p in parameters if (m := PARAMETER.fullmatch(p[0]))}
    if actual != expected:
        raise ValueError(f"Incomplete element parameter set: {chemistry.name}")
    analysis.update(Parameters=parameters, Number_of_samples=len(names),
                    Sample_list=names, Spectrum_dir=(material_dir / "data").resolve().as_posix(),
                    Result_dir=(material_dir / "fit").resolve().as_posix(), Restart_mode="New")
    return doc


def load_source(template: dict, source: Path) -> list[dict]:
    """Validate exact raster coverage and partition the original y bands."""
    files = sorted(source.glob("*_RBS.dat"))
    grid_x, grid_y = scan.array_scan_grid()
    expected = set(zip(np.rint(grid_x.ravel()*1e6).astype(np.int64),
                       np.rint(grid_y.ravel()*1e6).astype(np.int64)))
    actual = set()
    records = []
    unique_y = np.unique(np.rint(grid_y*1e6).astype(np.int64))[::-1]
    if len(unique_y) % scan.ARRAY_ROWS:
        raise ValueError("Raster y rows cannot be divided into equal material bands")
    rows_per_material = len(unique_y) // scan.ARRAY_ROWS
    y_to_row = {int(y): i // rows_per_material for i, y in enumerate(unique_y)}
    prefixes = set()
    for path in files:
        name = path.stem.removesuffix("_RBS")
        match = SAMPLE.fullmatch(name)
        if match is None:
            raise ValueError(f"Invalid spectrum name: {path.name}")
        prefix, x, theta1, y, theta2 = match.groups()
        if (int(theta1), int(theta2)) != (scan.THETA1, scan.THETA2):
            raise ValueError(f"Unexpected detector angles: {path.name}")
        x_nm, y_nm = int(x), int(y)
        if (x_nm, y_nm) not in expected or (x_nm, y_nm) in actual:
            raise ValueError(f"Unexpected/duplicate coordinate: {path.name}")
        actual.add((x_nm, y_nm))
        prefixes.add(prefix)
        row = y_to_row[y_nm]
        x_mm, y_mm = x_nm/1e6, y_nm/1e6
        cell_id = int(scan._array_cell_ids(np.array(x_mm), np.array(y_mm)))
        if cell_id >= 0 and scan._ARRAY_CELLS[cell_id]["row"] != row:
            raise ValueError("A material band intersects another material's electrode")
        records.append(dict(path=path, sample=name, row=row, cell_id=cell_id,
                            x_mm=x_mm, y_mm=y_mm))
    if actual != expected or len(prefixes) != 1:
        raise ValueError(f"Expected complete 100 x 160 raster; found {len(actual)} positions")
    configured_names = template["Config"]["Analysis"]["Sample_list"]
    if len(configured_names) != len(records) or set(configured_names) != {r["sample"] for r in records}:
        raise ValueError("Source spectra do not match the fit template's sample list")
    return records


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def prepare(template_path: Path, source: Path | None, output: Path, *, dry_run=False) -> list[dict]:
    with template_path.open("rb") as stream:
        template = tomllib.load(stream)
    source = source or Path(template["Config"]["Analysis"]["Spectrum_dir"])
    # Preflight every dependency before creating output or copying spectra.
    for method in template["Config"]["IBA"]["Methods"].values():
        if method["Simulator"] != "Surrogate":
            raise ValueError("Expected a Surrogate template")
        if not Path(method["Reference_file"]).is_file():
            raise FileNotFoundError(method["Reference_file"])
    if set(MATERIAL_FOLDERS) != {c.name for c in scan.CHEMISTRIES}:
        raise ValueError("Material names no longer match the generator")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    records = load_source(template, source)
    plans = []
    for row, chem in enumerate(scan.CHEMISTRIES):
        folder = MATERIAL_FOLDERS[chem.name]
        selected = [r for r in records if r["row"] == row]
        names = [r["sample"] for r in selected]
        directory = output / folder
        config = material_configuration(template, chem, names, directory)
        # TOML roundtrip before any write; only settings, never old fit results.
        if tomllib.loads(tomli_w.dumps(config)) != config:
            raise ValueError("AutoNRA TOML roundtrip failed")
        free = [p[0] for p in config["Config"]["Analysis"]["Parameters"] if p[-1] == "FREE"]
        plans.append(dict(chemistry=chem, folder=folder, records=selected, config=config, free=free))
        print(f"{folder}: {len(selected)} spectra, five SOC disks, {len(free)} free parameters; "
              f"elements {', '.join(fit_elements(chem))}", flush=True)
    if dry_run:
        return plans
    output.mkdir(parents=True, exist_ok=False)
    template_hash = hashlib.sha256(template_path.read_bytes()).hexdigest()
    generator_hash = hashlib.sha256(Path(scan.__file__).read_bytes()).hexdigest()
    thicknesses = scan.sublayer_thicknesses()
    summary = []
    for plan in plans:
        chem, selected = plan["chemistry"], plan["records"]
        directory = output / plan["folder"]
        (directory / "data").mkdir(parents=True)
        (directory / "fit").mkdir()
        fit_path = directory / "fit" / f"surr_{plan['folder']}.anra"
        fit_path.write_text(tomli_w.dumps(plan["config"]), encoding="utf-8")
        manifest = []
        # Evaluate the complete band together instead of traversing the full
        # forty-disk model once per spectrum.
        coordinates = np.array([(r["x_mm"], r["y_mm"]) for r in selected])
        depth_edges = np.r_[0, np.cumsum(thicknesses)]
        midpoints = (depth_edges[:-1] + depth_edges[1:]) / 2
        fractions = scan.array_composition(coordinates[:, 0, None], coordinates[:, 1, None],
                                           midpoints[None, :])
        inventories = np.einsum("l,nle->ne", thicknesses, fractions)
        for record, inventory in zip(selected, inventories, strict=True):
            data = record["path"].read_bytes()
            (directory / "data" / record["path"].name).write_bytes(data)
            cell_id = record["cell_id"]
            cell = scan._ARRAY_CELLS[cell_id] if cell_id >= 0 else None
            mean_ion = inventory[scan.ARRAY_ELEMENTS.index(chem.mobile_ion)] / thicknesses.sum()
            truth_soc = float(scan.soc_from_mobile_fraction(chem, mean_ion)) if cell else None
            manifest.append(dict(sample=record["sample"], spectrum_file=record["path"].name,
                source_sha256=hashlib.sha256(data).hexdigest(), x_mm=record["x_mm"], y_mm=record["y_mm"],
                cell_id=cell_id, region="electrode" if cell else "Al_holder",
                column=cell["column"]+1 if cell else "", nominal_soc_pct=100*cell["soc"] if cell else "",
                truth_depth_integrated_soc_pct=100*truth_soc if cell else "",
                **{f"truth_{e}_atomic_fraction": float(v/thicknesses.sum())
                   for e, v in zip(scan.ARRAY_ELEMENTS, inventory)}))
        write_csv(directory / "spectrum_manifest.csv", manifest)
        selected_cells = [c for c in scan.array_cell_layout() if c["row"] == selected[0]["row"]]
        metadata = dict(schema_version=1, chemistry=asdict(chem), elements=fit_elements(chem),
            sources={key: scan.SOURCES[key] for key in chem.source_keys},
            cells=selected_cells, scan_columns=100, scan_rows=20, number_of_spectra=len(selected),
            source_data=source.resolve().as_posix(), template=template_path.resolve().as_posix(),
            template_sha256=template_hash, generator_sha256=generator_hash,
            generation="Byte-for-byte subset of existing synthetic raster; no new spectra/noise generated",
            coordinates="Original full-array coordinates retained; one y band per material",
            ground_truth_assumption="Current generator matches original data; original scan metadata unavailable",
            thicknesses=thicknesses.tolist(), free_parameters=plan["free"],
            fit_settings="Original optimizer, objective, ROI weights, budget, bounds and initial values retained; inactive elements removed",
            anra=fit_path.resolve().as_posix())
        (directory / "scan_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
        summary.append(dict(material=chem.name, folder=plan["folder"], spectra=len(selected),
            electrode_positions=sum(r["cell_id"] >= 0 for r in selected),
            holder_positions=sum(r["cell_id"] < 0 for r in selected),
            free_parameters=len(plan["free"]), elements=", ".join(fit_elements(chem)),
            anra=f"{plan['folder']}/fit/surr_{plan['folder']}.anra"))
        print(f"Prepared {fit_path}", flush=True)
    write_csv(output / "materials.csv", summary)
    lines = ["# Material-specific surrogate fits", "",
        "Each folder contains five synthetic electrode disks at 0, 25, 50, 75 and 100% SOC.",
        "The original 100 x 160 raster is partitioned into eight 100 x 20 material bands.",
        "Spectra are copied byte for byte, preserving counting noise, acquisition times and coordinate names.",
        "Al holder positions are retained; each fit permits only that material's elements plus Al.",
        "", "Open each fit/surr_<material>.anra in AutoNRA. Fits have not been run.",
        "The surrogate model path, ARRDE settings, 10,000-evaluation budget, initial values, objective/ROI weights,",
        "seven logarithmic sublayers, fixed thickness and fixed instrument settings come from the working template.",
        "Only Li (lithium materials) or Na (sodium materials) has free depth-shape coefficients.",
        "Host/Al amplitudes remain independent; no true SOC or fixed host ratios are supplied to the optimizer.",
        "Other elements have been removed from both ElementsPerLayer and the parameter list.",
        "", "Inspect spectrum_manifest.csv for ground truth and source SHA256 values.",
        "Truth is reconstructed with the current generator; its equivalence to the original chemistry is assumed.",
        "The data are copied independently; changing a subset spectrum does not alter the source dataset.",
        "All paths in .anra files are absolute for this workstation.", "",
        "| Material | Spectra | Electrode | Holder | Free parameters | Elements |",
        "|---|---:|---:|---:|---:|---|"]
    lines.extend(f"| {r['material']} | {r['spectra']} | {r['electrode_positions']} | {r['holder_positions']} | {r['free_parameters']} | {r['elements']} |" for r in summary)
    (output / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return plans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=Path("E:/Test_data/Batt/LRN_ad/ref_lrn.anra"))
    parser.add_argument("--source-data", type=Path, help="defaults to the template's Spectrum_dir")
    parser.add_argument("--output", type=Path, default=scan.ROOT / "examples/datasets/surr_batt")
    parser.add_argument("--dry-run", action="store_true", help="validate and report without writing")
    args = parser.parse_args()
    prepare(args.template, args.source_data, args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
