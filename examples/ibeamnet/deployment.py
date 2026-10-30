"""One variable-layer RecurrentSurrogate package with explicit physical-AD preprocessing."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import tempfile
import numpy as np
import torch
import ibeamlab as ibl
from ibeamlab.export import validate_onnx_contract
from .data import digest, reference_from_dict, write_json
from .validation import validate_checkpoint_contract, validate_pileup
from .workflow import load_cache
from .training import load_checkpoint, make_model
from .onnx_export import export_graph


def detector_export_settings(manifest, settings):
    """Resolve physical export controls independently of training transforms."""
    overrides = settings.get("output_preparation", {})
    labels = {s["measurement"] for s in manifest["spectra"]}
    if not isinstance(overrides, dict) or set(overrides) - labels:
        raise ValueError("output_preparation must map known detector labels to controls")
    spectra, pileup = [], dict(settings.get("pileup", {}))
    for spec in manifest["spectra"]:
        label = spec["measurement"]
        control = overrides.get(label, {})
        if not isinstance(control, dict) or set(control) - {"rebin", "exposure_scale", "pileup"}:
            raise ValueError(f"output_preparation.{label}: unknown physical export controls")
        resolved = dict(spec)
        for name in ("rebin", "exposure_scale"):
            if name in control:
                if type(control[name]) is not bool:
                    raise ValueError(f"output_preparation.{label}.{name} must be boolean")
                resolved[name] = control[name]
        if "pileup" in control:
            pileup[label] = control["pileup"]
        spectra.append(resolved)
    validate_pileup(pileup, spectra)
    return spectra, pileup


def output_preparation(manifest, settings):
    """Build physical detector corrections: rebin, exposure, requested-time pileup.

    Training targets stay bare. Settings affect only exported physical inference.
    Where generation provenance is available, verify its reference checksum and
    disabled pileup flag rather than relying on the reference filename.
    """
    spectra, settings = detector_export_settings(manifest, settings)
    enabled = {label for label, item in settings.items() if item["enabled"]}
    roots = {Path(source["path"]).parent for source in manifest["sources"]}
    for root in roots:
        provenance = root / "generation.resolved.json"
        if not enabled or not provenance.is_file():
            continue
        recorded = json.loads(provenance.read_text(encoding="utf-8"))
        if recorded.get("backend") != "simnra":
            continue
        methods = {method["measurement"]: method for method in recorded["methods"]}
        for label in enabled:
            if label not in methods:
                raise ValueError(f"missing SIMNRA reference provenance for {label}")
            method = methods[label]
            reference = Path(method["reference"])
            if digest(reference) != method["sha256"]:
                raise ValueError(f"SIMNRA reference changed since generation: {reference}")
            flags = [node.text.strip().lower() for node in ET.parse(reference).getroot().iter()
                     if node.tag.split("}")[-1].lower() == "pileup" and node.text]
            if not flags or any(flag != "false" for flag in flags):
                raise ValueError(f"pileup-free targets require disabled SIMNRA pileup: {reference}")
    pipelines = []
    for spec in spectra:
        label = spec["measurement"]
        steps = []
        if spec["rebin"]:
            steps.append(ibl.Rebin())
        if spec["exposure_scale"]:
            steps.append(ibl.ExposureScale())
        if label in enabled:
            steps.append(ibl.Pileup(settings[label]["tau_seconds"], ibl.AcquisitionSource.REQUESTED))
        pipelines.append(ibl.SpectrumProcessing(label, steps))
    return pipelines


def output_transforms(manifest):
    """Undo the exact count scaling used to build training targets, per detector."""
    return [ibl.SpectrumTransform(spec["measurement"], ibl.TransformPipeline(
        [ibl.MultiplyScale([1. / spec["count_scale"]])], direction=ibl.TransformDirection.INVERSE))
        for spec in manifest["spectra"]]


def verify_physical_population(native, physical, expected, manifest, setup, pileup):
    """Check packaged pileup at reference acquisition times against NumPy FFT."""
    corrections = {}
    offset = 0
    for spec in manifest["spectra"]:
        label, count = spec["measurement"], spec["channels"]
        values = expected[:, offset:offset + count].astype(np.float64) * spec["count_scale"]
        offset += count
        measurement = next(m for m in setup.measurements if m.label == label)
        settings = pileup.get(label, {})
        if settings.get("enabled", False):
            acquisition = measurement.acquisition
            if not 0 < acquisition.live_time <= acquisition.real_time:
                raise ValueError(f"{label}: pileup requires positive real/live times, live <= real")
            tau = settings["tau_seconds"]
            total = values.sum(axis=1, keepdims=True)
            real, live = acquisition.real_time, acquisition.live_time
            width = 2 * count - 1
            nfft = 1 << (width - 1).bit_length()
            convolution = np.fft.irfft(np.fft.rfft(values, n=nfft, axis=1) ** 2, n=nfft, axis=1)[:, :width]
            factor = tau / real * np.exp(-tau * total / real)
            padded = np.pad(values, ((0, 0), (0, count - 1)))
            values = live / real * ((1 - 2 * factor * total) * padded + factor * np.maximum(convolution, 0))
        corrections[label] = values
    actual = native.predict_areal_density_batch(physical, setup=setup)
    for label, values in corrections.items():
        np.testing.assert_allclose(actual[label], values, rtol=3e-4, atol=2e-4)
    for row in range(len(physical)):
        result = native.predict_areal_density(physical[row], setup=setup)
        for label, values in corrections.items():
            np.testing.assert_allclose(result[label].counts, values[row], rtol=3e-4, atol=2e-4)


def build_export_metadata(manifest, settings, *, provenance=""):
    """Expose all four stages, using the exact fitted training/cache coefficients.

    Physical corrections are export controls; numerical transforms must retain
    the coefficients used to train this network.
    """
    reference = reference_from_dict(manifest["sources"][0]["reference"])
    return ibl.ModelMetadata(ibl.ForwardModelMetadata(
        reference_setup=reference.setup,
        input_preparation=ibl.LayerInputSchema(manifest["elements"]),
        input_transform=ibl.TransformPipeline([ibl.MultiplyScale([manifest["ad_input_factor"]])]),
        output_spectra=ibl.SpectraSchema([
            ibl.SpectrumSpec(s["measurement"], s["channels"]) for s in manifest["spectra"]]),
        output_transform=output_transforms(manifest),
        output_preparation=output_preparation(manifest, settings)), provenance=provenance)


def export(config, checkpoint=None):
    """Verify variable layer/batch counts before publishing one format-1 package.

    Graph inputs are scaled AD. Runtime packing and metadata apply physical
    conversion and scaling exactly once. Training counts are provenance only.
    """
    import onnx
    import onnxruntime as ort
    cache = Path(config["data"]["cache"])
    manifest = load_cache(config)
    checkpoint = Path(checkpoint or config["training"]["output"] / "best.pt")
    state = load_checkpoint(checkpoint)
    validate_checkpoint_contract(state, manifest, cache_sha256=digest(cache / "manifest.json"))
    network = make_model(manifest, state["model_options"]).eval()
    network.load_state_dict(state["model_state"])
    output = Path(config["export"]["output"])
    if output.exists():
        raise FileExistsError(output)
    factor = manifest["ad_input_factor"]
    reference = reference_from_dict(manifest["sources"][0]["reference"])
    trained_counts = [s["layer_count"] for s in manifest["sources"]]
    metadata = build_export_metadata(manifest, config["export"],
        provenance=f"RecurrentSurrogate; training_layer_counts={trained_counts}; checkpoint_sha256={digest(checkpoint)}")
    metadata.validate()
    _, pileup_settings = detector_export_settings(manifest, config["export"])
    reports = []
    with tempfile.TemporaryDirectory(prefix="ibeamnet-") as directory:
        graph = Path(directory) / "model.onnx"
        with torch.no_grad():
            export_graph(network, None, graph, dict(input_names=["inputs"], output_names=["outputs"],
                                                    opset_version=config["export"]["opset"]))
        validate_onnx_contract(onnx.load(graph), metadata)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"])
        package = ibl.Package.from_onnx(graph, metadata)
        native = ibl.ForwardModel(package) if config["export"]["verify_native"] else None
        rng = np.random.default_rng(42)
        for count in sorted({1, 2, 8, 17, manifest["maximum_layers"] + 3}):
            for batch in (1, 3):
                physical = rng.uniform(0, 50000, (batch, count, len(manifest["elements"]))).astype(np.float32)
                if count > 2:
                    physical[:, 1] = 0
                    physical[:, -1] = 0
                scaled = physical * np.float32(factor)
                with torch.no_grad():
                    expected = network(torch.from_numpy(scaled)).numpy()
                actual = session.run(None, {"inputs": scaled})[0]
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
                if native:
                    from ibeamlab._engine import engine
                    np.testing.assert_allclose(engine().run_graph(package.to_core(), scaled), expected, rtol=1e-4, atol=1e-5)
                    population = native.predict_reference_areal_density_batch(physical)
                    offset = 0
                    for spec in manifest["spectra"]:
                        stop = offset + spec["channels"]
                        np.testing.assert_allclose(population[spec["measurement"]],
                            expected[:, offset:stop] * spec["count_scale"], rtol=2e-4, atol=1e-4)
                        offset = stop
                    for row in range(batch):
                        predicted = native.predict_reference_areal_density(physical[row])
                        offset = 0
                        for spec in manifest["spectra"]:
                            stop = offset + spec["channels"]
                            np.testing.assert_allclose(predicted[spec["measurement"]].counts,
                                expected[row, offset:stop] * spec["count_scale"], rtol=2e-4, atol=1e-4)
                            offset = stop
                    verify_physical_population(native, physical, expected, manifest, reference.setup,
                                               pileup_settings)
                reports.append({"layers": count, "batch": batch,
                    "onnx_max_absolute_error": float(np.max(np.abs(actual - expected)))})
        empty = np.zeros((3, 5, len(manifest["elements"])), dtype=np.float32)
        with torch.no_grad():
            np.testing.assert_allclose(session.run(None, {"inputs": empty})[0],
                                       network(torch.from_numpy(empty)).numpy(), rtol=1e-4, atol=1e-5)
        output.mkdir(parents=True, exist_ok=False)
        package.write(output / "forward.ibm")
    write_json(output / "manifest.json", {"format": "ibeamlab.forward-export", "format_version": 1,
        "architecture": "RecurrentSurrogate", "ad_input_factor": factor, "package": "forward.ibm",
        "training_layer_counts": trained_counts, "native_verified": native is not None,
        "checkpoint_sha256": digest(checkpoint), "pileup": pileup_settings,
        "checks": reports})
    return reports
