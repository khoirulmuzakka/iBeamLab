"""Small end-to-end forward-surrogate checks; never uses SIMNRA or production data."""
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch
from copy import deepcopy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NATIVE = False
if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
    sys.path.insert(0, str(Path(sys.argv.pop(1)).resolve()))
    if len(sys.argv) > 1 and sys.argv[1] in ("ON", "OFF"):
        NATIVE = sys.argv.pop(1) == "ON"

import torch
import ibeamlab as ibl
from examples.ibeamnet.configuration import load_config
from examples.ibeamnet.validation import validate_schedule, validate_config, validate_model_options
from examples.ibeamnet.data import prepare
from examples.ibeamnet.model import RecurrentSurrogate, SpectrumLoss
from examples.ibeamnet.training import train, load_checkpoint, scheduled_epochs, atomic_checkpoint
from examples.ibeamnet.deployment import export

torch.set_num_threads(1)


class SurrogateTests(unittest.TestCase):
    def test_shared_validation_reports_override_fields(self):
        config = load_config(ROOT / "examples/ibeamnet/surrogate.toml")
        for section, key, invalid, message in (
            ("training", "gradient_clip", "bad", r"training.gradient_clip"),
            ("model", "hidden_size", True, r"model.hidden_size"),
            ("loss", "noise_floor", float("nan"), r"loss.noise_floor"),
            ("data", "layer_counts", [[1]], r"data.layer_counts\[0\]"),
        ):
            changed = deepcopy(config)
            changed[section][key] = invalid
            with self.assertRaisesRegex(ValueError, message):
                validate_config(changed)
        del config["training"]["batch_size"]
        with self.assertRaisesRegex(ValueError, r"training.batch_size is required"):
            validate_config(config)
        with self.assertRaisesRegex(ValueError, r"model.block_hidden_sizes\[0\]"):
            validate_model_options({"block_hidden_sizes": [0]})
        with self.assertRaisesRegex(ValueError, r"model.block_hidden_sizes\[0\]"):
            RecurrentSurrogate(["Si"], [8], block_hidden_sizes=[0])

    def test_staged_schedule_boundaries_and_validation(self):
        schedule = load_config(ROOT / "examples/ibeamnet/surrogate.toml")["training"]["schedule"]
        self.assertEqual(validate_schedule(schedule), 165)
        epochs = list(scheduled_epochs(schedule))
        for index, stage, rate in ((0, 1, 5e-4), (24, 1, 5e-4), (25, 2, 1e-4),
                                   (114, 2, 1e-4), (115, 3, 1e-5), (164, 3, 1e-5)):
            self.assertEqual(epochs[index], (index, stage, rate))
        self.assertEqual(list(scheduled_epochs(schedule, 65)), epochs[65:])
        self.assertEqual(list(scheduled_epochs(schedule, 165)), [])
        for invalid in ([], [{"epochs": 0, "learning_rate": .01}],
                        [{"epochs": True, "learning_rate": .01}],
                        [{"epochs": 1, "learning_rate": float("nan")}],
                        [{"epochs": 1, "learning_rate": -1}],
                        [{"epochs": 1, "learning_rate": True}]):
            with self.assertRaises(ValueError):
                validate_schedule(invalid)

    def test_default_architecture_matches_configuration(self):
        config = load_config(ROOT / "examples/ibeamnet/surrogate.toml")
        default = RecurrentSurrogate(["Si", "O"], [8])
        configured = RecurrentSurrogate(["Si", "O"], [8], **config["model"])
        self.assertEqual({k: tuple(v.shape) for k, v in default.state_dict().items()},
                         {k: tuple(v.shape) for k, v in configured.state_dict().items()})
        decoder = [m for m in default.heads[0].decoder if isinstance(m, torch.nn.Linear)]
        self.assertEqual([m.out_features for m in decoder], [512, 512, 8])
        self.assertEqual(default.heads[0].refiner.in_conv.kernel_size, (9,))
        # Both an empty decoder tower and different hidden widths are supported.
        direct = RecurrentSurrogate(["Si", "O"], [8], decoder_hidden_sizes=[], refiner_channels=0)
        self.assertEqual(len(direct.heads[0].decoder), 1)

    def test_dynamic_export_callback_and_schema_validation(self):
        import onnx
        import onnxruntime as ort
        from examples.ibeamnet.onnx_export import export_graph
        from ibeamlab.export import validate_onnx_contract
        network = RecurrentSurrogate(["Si", "O"], [8], hidden_size=8, embedding_size=8,
            contribution_size=8, decoder_hidden_sizes=[8, 8], refiner_channels=0,
            setup_embedding_size=4, block_hidden_sizes=[8]).eval()
        setup = ibl.ExperimentalSetup([ibl.Measurement("RBS", ibl.Beam("H", 2000))])
        metadata = ibl.ModelMetadata(ibl.ForwardModelMetadata(setup, ibl.LayerInputSchema(["Si", "O"],
            [ibl.MultiplyScale([1e-5])]), ibl.SpectraSchema([ibl.SpectrumSpec("RBS", 8)])))
        for opset in (17, 18):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                def callback(model, example, destination, options):
                    export_graph(model, example, destination, options)
                    graph = onnx.load(destination)
                    self.assertTrue(any(n.op_type == "Loop" for n in graph.graph.node))
                    self.assertEqual(graph.graph.input[0].type.tensor_type.shape.dim[1].dim_param, "layers")
                    onnx.save(graph, path / "graph.onnx")
                package = ibl.export_package(network, torch.ones(2, 1, 2), metadata, path / "forward.ibm",
                    opset=opset, verify=NATIVE, graph_exporter=callback)
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                session = ort.InferenceSession(str(path / "graph.onnx"), options, providers=["CPUExecutionProvider"])
                for layers in (1, 9, 32):
                    values = torch.rand(3, layers, 2)
                    values[:, 0] = 0
                    if layers > 1:
                        values[0, 4:] = 0
                        values[1, 2:] = 0
                        values[2] = 0
                    with torch.no_grad():
                        expected = network(values).numpy()
                    np.testing.assert_allclose(session.run(None, {"inputs": values.numpy()})[0],
                                               expected, rtol=1e-4, atol=1e-5)
                graph = onnx.load(path / "graph.onnx")
                graph.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 1
                with self.assertRaisesRegex(ValueError, "dynamic batch and layer"):
                    validate_onnx_contract(graph, metadata)
                if NATIVE:
                    onnx.save(graph, path / "fixed.onnx")
                    invalid = ibl.Package.from_onnx(path / "fixed.onnx", metadata)
                    with self.assertRaisesRegex(ibl.InferenceError, "dynamic batch and layer"):
                        ibl.ForwardModel(invalid)

    def test_padding_and_gradients(self):
        model = RecurrentSurrogate(["Si", "O"], [8, 11], hidden_size=8,
            embedding_size=8, contribution_size=8, decoder_hidden_sizes=[16, 16], refiner_channels=2, refiner_kernel=3,
            setup_embedding_size=4, block_hidden_sizes=[8])
        x = torch.tensor([[[.15, .35], [.12, .08]]], requires_grad=True)
        y = model(x)
        self.assertEqual(tuple(y.shape), (1, 19))
        self.assertTrue((y >= 0).all())
        torch.testing.assert_close(model(torch.cat((x, torch.zeros(1, 3, 2)), 1)), y)
        torch.testing.assert_close(model(torch.cat((x[:, :1], torch.zeros(1, 1, 2), x[:, 1:]), 1)), y)
        torch.testing.assert_close(model(torch.zeros(1, 1, 2)), model(torch.zeros(1, 4, 2)))
        loss = SpectrumLoss([{"channels": 8, "loss_weight": 1}, {"channels": 11, "loss_weight": 2}])
        loss(y, torch.ones_like(y)).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0)

    def test_full_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "datasets").mkdir()
            config = load_config(ROOT / "examples/ibeamnet/surrogate.toml")
            config["data"].update(source=root / "datasets", cache=root / "cache", layer_counts=[1, 2])
            config["training"].update(output=root / "run", device="cpu", batch_size=8,
                                     schedule=[{"epochs": 2, "learning_rate": .001}])
            config["export"].update(output=root / "packages", verify_native=NATIVE)
            config["model"] = dict(hidden_size=8, embedding_size=8, contribution_size=8,
                decoder_hidden_sizes=[16, 16], refiner_channels=2, refiner_kernel=3,
                setup_embedding_size=4, block_hidden_sizes=[8])
            config["spectra"] = [dict(measurement=label, channels=channels, count_scale=scale,
                loss_weight=1., exposure_scale=True, rebin=True)
                for label, channels, scale in (("RBS", 8, 100.), ("NRA", 11, 25.))]
            for layers in (1, 2):
                reference = ibl.Experiment(ibl.Sample([ibl.Layer(100, {"Si": .5, "O": .5}) for _ in range(layers)]),
                    ibl.ExperimentalSetup([ibl.Measurement(label, ibl.Beam("H", energy),
                        acquisition=ibl.Acquisition(1000)) for label, energy in (("RBS", 2000), ("NRA", 3000))]))
                parameters, rows = [], []
                for layer in range(layers):
                    parameters.extend([ibl.ParameterSpec(f"t{layer}", ibl.LayerThickness(layer)),
                        ibl.ParameterSpec(f"si{layer}", ibl.SpeciesConcentration(layer, "Si")),
                        ibl.ParameterSpec(f"o{layer}", ibl.SpeciesConcentration(layer, "O"))])
                for i in range(24):
                    fraction = .2 + i * .02
                    rows.append([v for _ in range(layers) for v in (100 + i, fraction, 1 - fraction)])
                rows = np.array(rows)
                if layers == 2:
                    parameters, rows = parameters[::-1], rows[:, ::-1].copy()
                ibl.GenerationPlan(reference, parameters).generate(rows, ibl.DummySimulator(7),
                    config["data"]["source"] / f"layers_{layers:02d}", records_per_shard=9)
            config["data"]["layer_counts"] = [1, 2, 3]
            with self.assertRaisesRegex(ValueError, "missing or incomplete"):
                prepare(config)
            self.assertFalse((root / "cache").exists())
            config["data"]["layer_counts"] = [1, 2]
            manifest = prepare(config)
            self.assertEqual(manifest["records"], 48)
            features = np.load(root / "cache/features.npy", mmap_mode="r")
            self.assertEqual(features.shape, (48, 2, 2))
            np.testing.assert_array_equal(features[:24, 1], 0)
            np.testing.assert_allclose(features[24, 0], np.array([20., 80.]) * config["preprocessing"]["ad_input_factor"])
            targets = np.load(root / "cache/targets.npy", mmap_mode="r")
            np.testing.assert_array_equal(targets[:, 7], 0)
            np.testing.assert_array_equal(targets[:, 15:], 0)
            splits = [np.load(root / f"cache/{name}_indices.npy") for name in ("train", "validation", "test")]
            np.testing.assert_array_equal(np.sort(np.concatenate(splits)), np.arange(48))
            del features, targets
            result = train(config)
            self.assertTrue(np.isfinite(result["test"]["loss"]))
            # Extend the first stage: resume in its middle, then cross into stage 2.
            config["training"]["schedule"] = [{"epochs": 3, "learning_rate": .001},
                                               {"epochs": 2, "learning_rate": .0001}]
            train(config, resume=True)
            resumed = load_checkpoint(root / "run/last.pt")
            self.assertEqual(resumed["completed_epochs"], 5)
            self.assertEqual([h["learning_rate"] for h in resumed["history"]], [.001] * 3 + [.0001] * 2)
            self.assertEqual([h["stage"] for h in resumed["history"]], [1, 1, 1, 2, 2])
            control = deepcopy(config)
            control["training"]["output"] = root / "uninterrupted"
            train(control)
            uninterrupted = load_checkpoint(root / "uninterrupted/last.pt")
            for key in resumed["model_state"]:
                torch.testing.assert_close(resumed["model_state"][key], uninterrupted["model_state"][key], rtol=0, atol=0)
            # Append stage 3. An update followed by interruption must not publish a partial epoch.
            config["training"]["schedule"].append({"epochs": 1, "learning_rate": .00005})
            step = torch.optim.AdamW.step
            def interrupt(optimizer, *args, **kwargs):
                step(optimizer, *args, **kwargs)
                raise KeyboardInterrupt()
            with patch.object(torch.optim.AdamW, "step", interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    train(config, resume=True)
            self.assertEqual(load_checkpoint(root / "run/last.pt")["completed_epochs"], 5)
            train(config, resume=True)
            control["training"]["schedule"] = deepcopy(config["training"]["schedule"])
            train(control, resume=True)
            resumed = load_checkpoint(root / "run/last.pt")
            uninterrupted = load_checkpoint(root / "uninterrupted/last.pt")
            self.assertEqual(resumed["history"][-1]["learning_rate"], .00005)
            for key in resumed["model_state"]:
                torch.testing.assert_close(resumed["model_state"][key], uninterrupted["model_state"][key], rtol=0, atol=0)
            for state in resumed["optimizer_state"]["state"].values():
                self.assertEqual(int(state["step"]), 30)  # 6 epochs * 5 batches; AdamW was never reset.
            invalid_config = deepcopy(config)
            invalid_config["training"]["schedule"] = [{"epochs": 1, "learning_rate": .001}]
            with self.assertRaisesRegex(ValueError, "shorter"):
                train(invalid_config, resume=True)
            invalid_config = deepcopy(config)
            invalid_config["training"]["batch_size"] = 4
            with self.assertRaisesRegex(ValueError, "other than"):
                train(invalid_config, resume=True)
            # Plateaued validation must not skip the final low-rate stage or
            # replace the protected best with equally good/latest weights.
            plateau = deepcopy(config)
            plateau["training"].update(output=root / "plateau",
                schedule=[{"epochs": 2, "learning_rate": .001}, {"epochs": 1, "learning_rate": .0001}])
            with patch("examples.ibeamnet.training.evaluate", return_value={"loss": 1., "samples": 4, "relative_l1": {}}):
                train(plateau)
            self.assertEqual(load_checkpoint(root / "plateau/last.pt")["completed_epochs"], 3)
            self.assertEqual(load_checkpoint(root / "plateau/best.pt")["epoch"], -1)
            # Weights-only initialization is distinct from exact training resume.
            weights = load_checkpoint(root / "run/best.pt")
            weights["format"] = "ibeamlab.forward-weights"
            atomic_checkpoint(root / "initial.pt", weights)
            warm = deepcopy(config)
            warm["training"].update(output=root / "warm", schedule=[{"epochs": 1, "learning_rate": .0001}])
            with patch.object(torch.optim.AdamW, "step", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    train(warm, initial_weights=root / "initial.pt")
            initial = load_checkpoint(root / "warm/last.pt")
            self.assertEqual(initial["completed_epochs"], 0)
            self.assertEqual(initial["optimizer_state"]["state"], {})
            for key, value in weights["model_state"].items():
                torch.testing.assert_close(initial["model_state"][key], value, rtol=0, atol=0)
            train(warm, resume=True)
            for value in load_checkpoint(root / "warm/last.pt")["optimizer_state"]["state"].values():
                self.assertEqual(int(value["step"]), 5)
            with self.assertRaisesRegex(ValueError, "either"):
                train(warm, resume=True, initial_weights=root / "initial.pt")
            reports = export(config)
            self.assertTrue(any(r["layers"] > 2 for r in reports))
            self.assertEqual(len(list((root / "packages").glob("*.ibm"))), 1)
            package = ibl.Package.open(root / "packages/forward.ibm")
            self.assertEqual(package.metadata.input_width(), 2)
            self.assertEqual(package.metadata.input_rank(), 3)
            self.assertEqual(package.metadata.output_width(), 19)
            self.assertEqual(len(package.metadata.model.postprocessing), 2)
            self.assertEqual(package.metadata.model.input.elements, ["Si", "O"])
            self.assertEqual(package.metadata.model.input.preprocessing[0].factors, [config["preprocessing"]["ad_input_factor"]])
            self.assertIn("RecurrentSurrogate", package.metadata.provenance)
            if NATIVE:
                from dataclasses import replace
                from examples.ibeamnet.training import make_model
                from ibeamlab._engine import engine
                state = load_checkpoint(root / "run/best.pt")
                network = make_model(manifest, state["model_options"]).eval()
                network.load_state_dict(state["model_state"])
                model = ibl.ForwardModel(package)
                for layers in (1, 2, 8, 32):
                    reference = ibl.Experiment(ibl.Sample([ibl.Layer(50000, {"O": .75, "Si": .25})
                        for _ in range(layers)]), package.metadata.model.reference_setup)
                    physical = np.tile(np.array([[12500., 37500.]], dtype=np.float32), (layers, 1))
                    saved = physical.copy()
                    original = model.predict(reference)
                    direct = model.predict_areal_density(physical)
                    np.testing.assert_array_equal(physical, saved)
                    with torch.no_grad():
                        expected = network(torch.from_numpy((physical * config["preprocessing"]["ad_input_factor"])[None])).numpy()[0]
                    np.testing.assert_allclose(original["RBS"].counts, expected[:8] * 100, rtol=2e-4)
                    np.testing.assert_allclose(direct["NRA"].counts, original["NRA"].counts, rtol=2e-4)
                    measurements = reference.setup.measurements
                    changed = replace(reference, setup=ibl.ExperimentalSetup([
                        replace(m, acquisition=ibl.Acquisition(2000 if m.label == "RBS" else 500))
                        for m in measurements[::-1]]))
                    predicted = model.predict(changed)
                    np.testing.assert_allclose(predicted["RBS"].counts, original["RBS"].counts * 2)
                    np.testing.assert_allclose(predicted["NRA"].counts, original["NRA"].counts * .5)
                for invalid in (np.zeros((0, 2), np.float32), np.ones((2, 3), np.float32),
                                np.full((2, 2), -1, np.float32), np.full((2, 2), np.nan, np.float32)):
                    with self.assertRaises(ibl.InferenceError):
                        model.predict_areal_density(invalid)
                missing = ibl.Experiment(ibl.Sample([ibl.Layer(100, {"Si": 1})]), reference.setup)
                np.testing.assert_allclose(model.predict(missing)["RBS"].counts,
                    model.predict_areal_density(np.array([[100, 0]], np.float32))["RBS"].counts)
                unsupported = replace(missing, sample=ibl.Sample([ibl.Layer(100, {"Fe": 1})]))
                with self.assertRaises(ibl.InferenceError):
                    model.predict(unsupported)
            if NATIVE:
                import _ibeamlab
                subprocess.run([sys.executable, str(ROOT / "examples/ibeamnet/predict.py"),
                    str(root / "packages/forward.ibm"), str(root / "datasets/layers_02"),
                    "--extension-dir", str(Path(_ibeamlab.__file__).parent),
                    "--output", str(root / "prediction.npz")], check=True)
                with np.load(root / "prediction.npz") as arrays:
                    self.assertEqual(arrays["prediction_NRA"].shape, (11,))
            config["data"]["cache"] = root / "cropped-cache"
            config["spectra"][0]["channels"] = 1
            with self.assertRaisesRegex(ValueError, "nonzero spectrum tail"):
                prepare(config)
            self.assertFalse((root / "cropped-cache/manifest.json").exists())
            self.assertTrue((root / "cropped-cache/incomplete").exists())


if __name__ == "__main__":
    unittest.main()
