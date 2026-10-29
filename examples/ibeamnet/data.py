"""Stream completed core datasets into fixed-grid, memory-mapped training arrays."""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
import shutil
import numpy as np
from tqdm import tqdm
import ibeamlab as ibl
from ibeamlab._engine import engine
from .validation import validate_config


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def reference_from_dict(value):
    """Restore physical metadata without relying on archived bindings or pickled classes."""
    layers = []
    for layer in value["sample"]["layers"]:
        species = [ibl.Species(s["element"], s["concentration"],
            tuple(ibl.Isotope(**i) for i in s["isotopes"])) for s in layer["species"]]
        layers.append(ibl.Layer(layer["thickness"], species, roughness=layer["roughness"],
            porosity_fraction=layer["porosity_fraction"], pore_diameter=layer["pore_diameter"]))
    setup = ibl.ExperimentalSetup([ibl.Measurement(m["label"], ibl.Beam(**m["beam"]),
        ibl.Detector(**m["detector"]), ibl.Acquisition(**m["acquisition"]))
        for m in value["setup"]["measurements"]])
    return ibl.Experiment(ibl.Sample(layers), setup)


def parameter_layout(metadata, elements):
    """Map declared physical columns to [layer, thickness+ordered concentrations]."""
    core = engine()
    layers = len(metadata.reference.sample.layers)
    columns = np.full((layers, len(elements) + 1), -1, dtype=np.int64)
    parameters = []
    for index, parameter in enumerate(metadata.parameters):
        target = parameter.target
        if isinstance(target, core.LayerThickness):
            column, kind = 0, "thickness"
        elif isinstance(target, core.SpeciesConcentration):
            if target.element not in elements:
                raise ValueError("inconsistent element set between datasets")
            column, kind = 1 + elements.index(target.element), "concentration"
        else:
            raise ValueError("this example requires fixed setups and thickness/composition parameters only")
        if not 0 <= target.layer < layers or columns[target.layer, column] >= 0:
            raise ValueError("duplicate or invalid parameter target")
        columns[target.layer, column] = index
        parameters.append({"name": parameter.name, "unit": parameter.unit, "kind": kind,
                           "layer": target.layer, "element": elements[column - 1] if column else None})
    if np.any(columns < 0):
        raise ValueError("every layer requires thickness and every configured element concentration")
    return columns, parameters


def load_manifest(cache):
    cache = Path(cache)
    with (cache / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("format") != "ibeamlab.surrogate-cache" or not manifest.get("complete"):
        raise ValueError("training cache is not complete")
    features = np.load(cache / "features.npy", mmap_mode="r")
    targets = np.load(cache / "targets.npy", mmap_mode="r")
    if manifest.get("input_representation") != "scaled_elemental_areal_density":
        raise ValueError("cache must contain scaled elemental areal densities; prepare a new cache")
    expected_x = (manifest["records"], manifest["maximum_layers"], len(manifest["elements"]))
    expected_y = (manifest["records"], sum(s["channels"] for s in manifest["spectra"]))
    if features.shape != expected_x or targets.shape != expected_y or features.dtype != np.float32 or targets.dtype != np.float32:
        raise ValueError("cache array shape/type differs from manifest")
    return manifest


def prepare(config):
    """Prepare one scaled elemental-AD row per record; detectors share a split.

    Cache arrays are float32 and disk-backed. Spectra are padded with zeros to
    the configured grid. Nonzero tails are never silently cropped. Failed
    records are skipped; incomplete layer systems are rejected before allocation.
    """
    validate_config(config)
    settings, spectra = config["data"], config["spectra"]
    cache = Path(settings["cache"])
    sources, datasets, elements, reference_setup = [], [], None, None
    offset = 0
    for layer_count in settings["layer_counts"]:
        path = Path(settings["source"]) / f"layers_{layer_count:02d}"
        if not (path / "manifest.ibd").is_file():
            raise ValueError(f"{path} is missing or incomplete; wait for generation to finish")
        dataset = ibl.open_dataset(path)
        metadata = dataset.metadata
        if len(metadata.reference.sample.layers) != layer_count:
            raise ValueError(f"wrong layer count in {path}")
        for layer in metadata.reference.sample.layers:
            if (layer.roughness != 0 or layer.porosity_fraction != 0 or layer.pore_diameter != 0
                    or any(s.isotopes for s in layer.species)):
                raise ValueError("elemental AD models require natural elements without roughness/porosity")
        current = tuple(s.element for s in metadata.reference.sample.layers[0].species)
        if elements is None:
            elements = current
        if any(set(s.element for s in layer.species) != set(elements) for layer in metadata.reference.sample.layers):
            raise ValueError("all layer systems must use the same element set")
        setup = {m.label: m for m in metadata.reference.setup.measurements}
        if reference_setup is None:
            reference_setup = setup
        if setup != reference_setup or set(setup) != {s["measurement"] for s in spectra}:
            raise ValueError("all sources must share the configured fixed measurement setup")
        for spec in spectra:
            if spec["exposure_scale"] and setup[spec["measurement"]].acquisition.particles_sr <= 0:
                raise ValueError("exposure scaling requires positive reference particles_sr")
        columns, parameters = parameter_layout(metadata, elements)
        count = metadata.accepted
        if settings["max_records_per_source"]:
            count = min(count, settings["max_records_per_source"])
        if count < 3:
            raise ValueError("each layer system needs at least three successful records for train/validation/test")
        sources.append({"path": str(path.resolve()), "layer_count": layer_count,
            "start": offset, "records": count, "columns": columns.tolist(), "parameters": parameters,
            "reference": {"sample": asdict(metadata.reference.sample), "setup": asdict(metadata.reference.setup)},
            "provenance": metadata.provenance,
            "manifest_sha256": digest(path / "manifest.ibd")})
        datasets.append(dataset)
        offset += count
    cache.parent.mkdir(parents=True, exist_ok=True)
    maximum = max(settings["layer_counts"])
    required = offset * (4 * (maximum * len(elements) + sum(s["channels"] for s in spectra)) + 24)
    if shutil.disk_usage(cache.parent).free < required:
        raise ValueError(f"training cache requires at least {required / 1e9:.2f} GB of free disk space")
    cache.mkdir(parents=True, exist_ok=False)
    (cache / "incomplete").write_text("preparing\n")
    features = np.lib.format.open_memmap(cache / "features.npy", mode="w+", dtype=np.float32,
                                        shape=(offset, maximum, len(elements)))
    targets = np.lib.format.open_memmap(cache / "targets.npy", mode="w+", dtype=np.float32,
                                       shape=(offset, sum(s["channels"] for s in spectra)))
    original_indices = np.lib.format.open_memmap(cache / "sample_indices.npy", mode="w+", dtype=np.uint64, shape=(offset,))
    splits = {name: [] for name in ("train", "validation", "test")}
    with tqdm(total=offset, desc="Prepare surrogate cache", unit="samples") as progress:
        for source, dataset in zip(sources, datasets):
            columns = np.asarray(source["columns"], dtype=np.int64)
            used = 0
            for record in dataset:
                if record.result.failed:
                    continue
                index = source["start"] + used
                values = record.parameters[columns]
                if (not np.isfinite(values).all() or (values < 0).any()
                        or (values > np.finfo(np.float32).max).any()
                        or not np.allclose(values[:, 1:].sum(1), 1, rtol=0, atol=1e-6)):
                    raise ValueError(f"invalid physical parameters at {source['path']} row {record.sample_index}")
                features[index].fill(0)
                densities = values[:, :1] * values[:, 1:] * config["preprocessing"]["ad_input_factor"]
                if not np.isfinite(densities).all() or (densities > np.finfo(np.float32).max).any():
                    raise ValueError("scaled areal densities exceed float32 range")
                features[index, :source["layer_count"]] = densities
                targets[index].fill(0)
                channel = 0
                if set(record.result) != {s["measurement"] for s in spectra}:
                    raise ValueError("record spectrum labels differ from configured outputs")
                for spec in spectra:
                    counts = record.result[spec["measurement"]].counts
                    size = spec["channels"]
                    if len(counts) > size and np.any(counts[size:] != 0):
                        raise ValueError(f"nonzero spectrum tail would be cropped; increase {spec['measurement']} channels")
                    length = min(len(counts), size)
                    scaled = counts[:length] / spec["count_scale"]
                    if scaled.size and scaled.max() > np.finfo(np.float32).max:
                        raise ValueError("scaled targets exceed float32 range")
                    targets[index, channel:channel + length] = scaled
                    channel += size
                original_indices[index] = record.sample_index
                used += 1
                progress.update()
                if used == source["records"]:
                    break
            if used != source["records"]:
                raise ValueError("source record count differs from its metadata")
            order = np.random.default_rng(np.random.SeedSequence([settings["seed"], source["layer_count"]])).permutation(used)
            validation = max(1, int(used * settings["validation_fraction"]))
            test = max(1, int(used * settings["test_fraction"]))
            if validation + test >= used:
                raise ValueError("split fractions leave no training rows")
            splits["validation"].append(source["start"] + order[:validation])
            splits["test"].append(source["start"] + order[validation:validation + test])
            splits["train"].append(source["start"] + order[validation + test:])
    features.flush(); targets.flush(); original_indices.flush()
    for name, parts in splits.items():
        np.save(cache / f"{name}_indices.npy", np.concatenate(parts))
    manifest = {"format": "ibeamlab.surrogate-cache", "format_version": 1, "complete": True,
        "records": offset, "maximum_layers": maximum, "elements": list(elements),
        "spectra": spectra, "input_representation": "scaled_elemental_areal_density",
        "ad_input_factor": config["preprocessing"]["ad_input_factor"],
        "sources": sources, "split_seed": settings["seed"],
        "split_sizes": {name: sum(map(len, parts)) for name, parts in splits.items()}}
    write_json(cache / "manifest.tmp", manifest)
    (cache / "manifest.tmp").rename(cache / "manifest.json")
    (cache / "incomplete").unlink()
    return manifest
