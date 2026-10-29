"""One variable-layer RecurrentSurrogate package with explicit physical-AD preprocessing."""
from pathlib import Path
import tempfile
import numpy as np
import torch
import ibeamlab as ibl
from ibeamlab.export import validate_onnx_contract
from .data import digest, reference_from_dict, write_json
from .validation import validate_checkpoint_contract
from .workflow import load_cache
from .training import load_checkpoint, make_model
from .onnx_export import export_graph


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
    pipelines = []
    for spec in manifest["spectra"]:
        steps = [ibl.MultiplyCounts(spec["count_scale"])]
        if spec["exposure_scale"]:
            steps.append(ibl.ExposureScale())
        if spec["rebin"]:
            steps.append(ibl.Rebin())
        pipelines.append(ibl.SpectrumProcessing(spec["measurement"], steps))
    trained_counts = [s["layer_count"] for s in manifest["sources"]]
    metadata = ibl.ModelMetadata(ibl.ForwardModelMetadata(reference.setup,
        ibl.LayerInputSchema(manifest["elements"], preprocessing=[ibl.MultiplyScale([factor])]),
        ibl.SpectraSchema([ibl.SpectrumSpec(s["measurement"], s["channels"]) for s in manifest["spectra"]]),
        postprocessing=pipelines), provenance=f"RecurrentSurrogate; training_layer_counts={trained_counts}; checkpoint_sha256={digest(checkpoint)}")
    metadata.validate()
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
                    for row in range(batch):
                        predicted = native.predict_areal_density(physical[row])
                        offset = 0
                        for spec in manifest["spectra"]:
                            stop = offset + spec["channels"]
                            np.testing.assert_allclose(predicted[spec["measurement"]].counts,
                                expected[row, offset:stop] * spec["count_scale"], rtol=2e-4, atol=1e-4)
                            offset = stop
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
        "checkpoint_sha256": digest(checkpoint), "checks": reports})
    return reports
