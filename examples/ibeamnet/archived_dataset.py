"""Read archived dataset.toml/IBEAMDS1 sources for the surrogate example.

This adapter does not modify archives or require the archived extension. Shards
are memory-mapped and each record's parameter/count arrays borrow their bytes.
Only successful records are stored in these shards; failure logs are separate.
"""
from pathlib import Path
import struct
from types import SimpleNamespace
import numpy as np
import ibeamlab as ibl
from ibeamlab.datasets import DatasetMetadata, DatasetRecord


class _Spectra(dict):
    failed = False


class ArchivedDataset:
    """Read a completed archive, restoring geometry from its recorded XNRA file."""
    def __init__(self, path):
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        self.path = Path(path)
        with (self.path/"dataset.toml").open("rb") as stream:
            raw = tomllib.load(stream)
        if raw.get("format") != "ibeamlab.dataset" or raw.get("format_version") != 1 or raw.get("complete") is not True:
            raise ValueError(f"Invalid or incomplete archived dataset: {self.path}")
        generation = raw["generation"]
        layers = []
        for layer in generation["sample"]["layers"]:
            species = [ibl.Species(s["element"], s["concentration"],
                       tuple(ibl.Isotope(**i) for i in s.get("isotopes", []))) for s in layer["species"]]
            layers.append(ibl.Layer(layer["thickness"], species,
                roughness=layer.get("roughness", 0), porosity_fraction=layer.get("porosity_fraction", 0),
                pore_diameter=layer.get("pore_diameter", 0)))
        methods = {m["label"]: m for m in generation["methods"]}
        measurements = []
        for detector in generation["setup"]["detectors"]:
            label = detector["label"]
            reference = Path(methods[label]["reference_file"])
            if not reference.is_absolute():
                reference = self.path/reference
            geometry = ibl.SimnraMethod(label, reference, detector["particle"]).reference_geometry()
            measurements.append(ibl.Measurement(label,
                ibl.Beam(detector["particle"], detector["beam_energy"], detector["beam_spread"]),
                ibl.Detector(detector["calibration_linear"], detector["calibration_offset"],
                    detector["calibration_quadratic"], detector["resolution"], detector.get("info", ""), geometry),
                ibl.Acquisition(detector["particles_sr"], detector["real_time"], detector["live_time"])))
        parameters = []
        for parameter in generation["parameters"]:
            if not parameter.get("open", True):
                continue
            target = parameter["target"]
            if target["type"] == "layer_thickness":
                value = ibl.LayerThickness(target["layer"])
            elif target["type"] == "species_concentration":
                value = ibl.SpeciesConcentration(target["layer"], target["element"])
            else:
                raise ValueError(f"Unsupported archived parameter target: {target}")
            parameters.append(ibl.ParameterSpec(parameter["name"], value, parameter.get("unit", "")))
        if [p.name for p in parameters] != raw["parameter_names"]:
            raise ValueError("Archived parameter order differs from generation metadata")
        self.labels = tuple(raw["spectrum_labels"])
        self.widths = tuple(raw["spectrum_lengths"])
        if len(self.labels) != len(self.widths) or len(set(self.labels)) != len(self.labels) or any(n < 1 for n in self.widths):
            raise ValueError("Invalid archived spectrum schema")
        self.shards = [self.path/name for name in raw["shards"]]
        if not self.shards or any(p.parent.resolve() != self.path.resolve() or not p.is_file() for p in self.shards):
            raise ValueError("Missing or invalid archived shard")
        self.metadata = DatasetMetadata(ibl.Experiment(ibl.Sample(layers), ibl.ExperimentalSetup(measurements)),
            tuple(parameters), f"archived-dataset; source={self.path.resolve()}",
            raw["requested"], raw["accepted"], raw["failed"], raw["invalid"])

    def maximum_spectrum_lengths(self):
        """Use declared bounds; individual records are checked while streaming."""
        return dict(zip(self.labels, self.widths))

    def __iter__(self):
        records = 0
        for path in self.shards:
            data = np.memmap(path, mode="r", dtype=np.uint8)
            if bytes(data[:8]) != b"IBEAMDS1":
                raise ValueError(f"Invalid archived shard magic: {path}")
            position = 8

            def take(size):
                nonlocal position
                start = position
                position += size
                if size < 0 or position > data.size:
                    raise ValueError(f"Truncated archived shard: {path}")
                return start

            def integer(kind):
                return struct.unpack_from(kind, data, take(struct.calcsize(kind)))[0]

            def string():
                size = integer("<I")
                if size > 16*1024*1024:
                    raise ValueError("Archived string exceeds size limit")
                start = take(size)
                return bytes(data[start:start+size]).decode("utf-8")

            while position < data.size:
                index, sample_id = integer("<Q"), string()
                count = integer("<I")
                if count != len(self.metadata.parameters):
                    raise ValueError("Archived parameter width differs from metadata")
                parameters = np.frombuffer(data, dtype="<f8", count=count, offset=take(count*8))
                if integer("<I") != len(self.labels):
                    raise ValueError("Archived spectrum count differs from metadata")
                spectra = _Spectra()
                for label, width in zip(self.labels, self.widths):
                    if string() != label:
                        raise ValueError("Archived spectrum order differs from metadata")
                    length = integer("<Q")
                    if length > width:
                        raise ValueError("Archived spectrum exceeds its declared width")
                    counts = np.frombuffer(data, dtype="<f4", count=length, offset=take(length*4))
                    if not np.isfinite(counts).all() or (counts < 0).any():
                        raise ValueError("Invalid archived spectrum counts")
                    spectra[label] = SimpleNamespace(counts=counts)
                records += 1
                yield DatasetRecord(index, sample_id, parameters, spectra)
        if records != self.metadata.accepted:
            raise ValueError("Archived successful record count differs from metadata")
