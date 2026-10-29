"""Tests of the Python-facing package against the current C++ execution engine."""
from dataclasses import replace
from pathlib import Path
import gc
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ONNX = os.environ.get("IBEAMLAB_TEST_ONNX", "OFF") == "ON"
if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
    sys.path.insert(0, str(Path(sys.argv.pop(1)).resolve()))
    ONNX = sys.argv.pop(1) == "ON"
elif os.environ.get("IBEAMLAB_EXTENSION_DIR"):
    sys.path.insert(0, os.environ["IBEAMLAB_EXTENSION_DIR"])

import ibeamlab as ibl
from ibeamlab._engine import engine
from ibeamlab.sample import experiment_from_core
try:
    core = engine()
except ibl.NativeExtensionError:
    core = None


def experiment():
    return ibl.Experiment(ibl.Sample([ibl.Layer(100, {"Si": .5, "O": .5})]),
        ibl.ExperimentalSetup([
            ibl.Measurement("RBS", ibl.Beam("H", 2000), acquisition=ibl.Acquisition(1000, 10, 8)),
            ibl.Measurement("NRA", ibl.Beam("H", 3000), acquisition=ibl.Acquisition(2000, 10, 8)),
        ]))


class DomainTests(unittest.TestCase):
    @unittest.skipUnless(core is not None, "compiled extension required")
    def test_metadata_conversion_errors_include_field_paths(self):
        with self.assertRaisesRegex(ibl.ValidationError, r"LayerInputSchema.preprocessing"):
            ibl.LayerInputSchema(["Si"], preprocessing=[ibl.MultiplyCounts(2)]).to_core()
        with self.assertRaisesRegex(ibl.ValidationError, r"ForwardModelMetadata.input"):
            ibl.ForwardModelMetadata(experiment().setup, 123,
                ibl.SpectraSchema([ibl.SpectrumSpec("RBS", 4)])).to_core()
        with self.assertRaisesRegex(ibl.ValidationError, r"ParameterSpec.target"):
            ibl.ParameterSpec("bad", 123).to_core()
        with self.assertRaisesRegex(ibl.ValidationError, r"ParameterSpec.unit"):
            ibl.ParameterSpec("bad", ibl.LayerThickness(0), 123).to_core()
        invalid = ibl.ModelMetadata(ibl.InverseModelMetadata(experiment().setup,
            ibl.SpectraSchema([ibl.SpectrumSpec("RBS", 4), ibl.SpectrumSpec("NRA", 3)]), ["Si"], 0))
        with self.assertRaisesRegex(ibl.ValidationError, r"ModelMetadata:.*layers"):
            invalid.validate()

    def test_measurements_and_validation(self):
        x = experiment()
        self.assertEqual([m.beam.energy for m in x.setup.measurements], [2000, 3000])
        for make in [lambda: ibl.Beam("H", 0), lambda: ibl.Layer(10, {"Si": .4}),
                     lambda: ibl.Detector(calibration_offset=float("nan")),
                     lambda: ibl.Acquisition(real_time=-1),
                     lambda: ibl.ExperimentalSetup([x.setup.measurements[0]] * 2)]:
            with self.assertRaises(ibl.ValidationError):
                make()

    def test_import_is_lazy(self):
        code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); import ibeamlab; "
                "assert 'torch' not in sys.modules; assert '_ibeamlab' not in sys.modules; "
                "assert 'ibeamlab._ibeamlab' not in sys.modules; ibeamlab.Beam('H', 2000)")
        subprocess.run([sys.executable, "-I", "-c", code], check=True)

    def test_numpy_spectrum_and_duplicate_labels(self):
        array = np.ones(4, dtype=np.float64)
        spectrum = ibl.Spectrum("RBS", array)
        self.assertIs(spectrum.counts, array)
        self.assertEqual(ibl.Spectrum("NRA", [1, 2]).counts.dtype, np.float64)
        with self.assertRaises(ibl.ValidationError):
            ibl.SimulationResult([spectrum, spectrum])
        with self.assertRaises(ibl.ValidationError):
            ibl.Spectrum("RBS", [float("nan")])


@unittest.skipIf(core is None, "compiled extension unavailable")
class EngineTests(unittest.TestCase):
    def test_roundtrip_and_cached_materialization(self):
        x = experiment()
        self.assertEqual(experiment_from_core(x.to_core()), x)
        self.assertIs(x.to_core(), x.to_core())
        other = replace(x, sample=ibl.Sample([ibl.Layer(150, {"Si": 1})]))
        self.assertIsNot(other.to_core(), x.to_core())
        self.assertEqual(core.read_parameter(other.to_core(), ibl.LayerThickness(0)), 150)

    def test_processing_reuses_outputs(self):
        values = np.array([1., 2., 3., 4.])
        edges = np.arange(5, dtype=np.float64)
        output = np.empty(4)
        self.assertIs(ibl.rebin(values, edges, edges, output=output), output)
        np.testing.assert_array_equal(output, values)
        with self.assertRaises(ValueError):
            ibl.rebin(values, edges, edges, output=np.empty(4, dtype=np.float32))
        piled = ibl.apply_pileup(values, real_time=10, live_time=8, tau=.001,
                                 workspace=ibl.PileupWorkspace())
        restored = ibl.remove_pileup(piled, real_time=10, live_time=8, tau=.001)
        np.testing.assert_allclose(restored, values, rtol=1e-6)

    def test_result_array_ownership(self):
        result = ibl.DummySimulator(4).simulate(experiment())
        counts = result["RBS"].counts
        expected = counts.copy()
        self.assertFalse(counts.flags.writeable)
        self.assertIsNotNone(counts.base)
        del result
        gc.collect()
        np.testing.assert_array_equal(counts, expected)
        with self.assertRaises(ValueError):
            counts[0] = 0

    def test_generation_and_independent_dataset_cursors(self):
        plan = ibl.GenerationPlan(experiment(), [ibl.vary.layer_thickness(layer=0, bounds=(50, 150))])
        rows = plan.sample_uniform(3, seed=42)
        np.testing.assert_array_equal(rows, plan.sample_uniform(3, seed=42))
        sim = ibl.DummySimulator(4)
        self.assertEqual(sim.simulate(experiment())["NRA"].counts.shape, (4,))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset"
            summary = plan.generate(rows, sim, path, records_per_shard=1)
            self.assertEqual(summary.accepted, 3)
            dataset = ibl.open_dataset(path)
            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset.metadata.parameter_names, ("layer_0_thickness",))
            np.testing.assert_array_equal(dataset.parameters, rows)
            self.assertEqual(dataset.spectra("NRA").shape, (3, 4))
            first, second = iter(dataset), iter(dataset)
            self.assertEqual(next(first).sample_index, next(second).sample_index)
            self.assertEqual([len(b) for b in dataset.iter_batches(2)], [2, 1])
            with self.assertRaises(ibl.DatasetError):
                plan.generate(rows, sim, path)
            bad = np.array([[100.], [-1.], [120.]])
            summary = plan.generate(bad, sim, Path(tmp) / "failed", failure_policy=ibl.FailurePolicy.RECORD)
            self.assertEqual((summary.accepted, summary.failed), (2, 1))
            failed = ibl.Dataset(Path(tmp) / "failed")
            self.assertEqual((len(failed), len(failed.failures)), (3, 1))
            self.assertEqual(failed.parameters.shape, (2, 1))
            sim.reset_stop()
            summary = plan.generate(rows, sim, Path(tmp) / "cancelled",
                progress=lambda done, total: sim.request_stop() if done == 1 else None)
            self.assertTrue(summary.cancelled)
            with self.assertRaises(ibl.DatasetError):
                ibl.Dataset(Path(tmp) / "cancelled")


@unittest.skipUnless(core is not None and ONNX, "ONNX-enabled extension required")
class ModelTests(unittest.TestCase):
    def test_multidetector_packages_and_inference(self):
        x = experiment()
        schema = ibl.SpectraSchema([ibl.SpectrumSpec("RBS", 4), ibl.SpectrumSpec("NRA", 3)])
        forward = ibl.ForwardModelMetadata(x.setup, ibl.ParameterInputSchema(x.sample,
            [ibl.ParameterSpec("thickness", ibl.LayerThickness(0))]), schema,
            postprocessing=[ibl.SpectrumProcessing("RBS", [ibl.ExposureScale()]),
                            ibl.SpectrumProcessing("NRA", [ibl.MultiplyCounts(.5), ibl.ExposureScale()])])
        metadata = ibl.ModelMetadata(forward, provenance="python-wrapper-test")
        metadata.validate()
        self.assertEqual((metadata.input_width(), metadata.output_width()), (1, 7))
        reference = x.setup.measurements
        requested = ibl.ExperimentalSetup([
            replace(reference[1], acquisition=ibl.Acquisition(1000, 10, 8)),
            replace(reference[0], acquisition=ibl.Acquisition(2000, 10, 8))])
        fixtures = ROOT / "cpp/tests/fixtures"
        with tempfile.TemporaryDirectory() as tmp:
            package = ibl.Package.from_onnx(fixtures / "forward_multi.onnx", metadata)
            path = Path(tmp) / "forward.ibm"
            package.write(path)
            model = ibl.load_model(path)
            self.assertIsInstance(model, ibl.ForwardModel)
            self.assertEqual(model.metadata.model.reference_setup, x.setup)
            self.assertEqual(model.metadata.model.input.reference_sample, x.sample)
            predicted = model.predict(replace(x, setup=requested))
            np.testing.assert_allclose(predicted["RBS"].counts, [200, 400, 600, 800])
            np.testing.assert_allclose(predicted["NRA"].counts, [125, 150, 175])
            self.assertEqual(len(model.predict_batch([x, x])), 2)
            inverse = ibl.ModelMetadata(ibl.InverseModelMetadata(x.setup, schema, ["Si", "O"], 1, True,
                input_preparation=[ibl.SpectrumProcessing(label, [ibl.ExposureScale()]) for label in ("RBS", "NRA")]))
            inverse_package = ibl.Package.from_onnx(fixtures / "inverse_multi.onnx", inverse)
            inverse_path = Path(tmp) / "inverse.ibm"
            inverse_package.write(inverse_path)
            inverse_model = ibl.load_model(inverse_path)
            self.assertIsInstance(inverse_model, ibl.InverseModel)
            measured = ibl.InverseInput(requested, {"NRA": [.5, 1, 1.5], "RBS": [2, 4, 6, 8]})
            prediction = inverse_model.predict(measured)
            np.testing.assert_allclose(prediction.values, [[70, 140]])
            np.testing.assert_allclose(prediction.presence_probability, [[.8, .5]])
            self.assertEqual(prediction.posterior_std.shape, (1, 2))
            retained = prediction.values
            self.assertFalse(retained.flags.writeable)
            del prediction
            gc.collect()
            np.testing.assert_allclose(retained, [[70, 140]])
            prepared = inverse_model.predict_prepared({"RBS": [1, 2, 3, 4], "NRA": [1, 2, 3]})
            np.testing.assert_allclose(prepared.values, retained)
            with self.assertRaises(ibl.InferenceError):
                inverse_model.predict(ibl.InverseInput(requested, {"wrong": [1, 2, 3]}))

    def test_optional_training_and_export(self):
        import torch
        from ibeamlab.training import fit
        torch.manual_seed(42)
        model = torch.nn.Linear(1, 4)
        inputs = np.arange(1, 11, dtype=np.float32).reshape(-1, 1) / 10
        targets = inputs * np.arange(1, 5, dtype=np.float32)
        model.eval()
        history = fit(model, inputs, targets, epochs=50, batch_size=5, learning_rate=.05,
                      validation_data=(inputs, targets))
        self.assertFalse(model.training)
        self.assertLess(history.loss[-1], history.loss[0])
        self.assertEqual(len(history.validation_loss), 50)
        x = experiment()
        x = replace(x, setup=ibl.ExperimentalSetup([x.setup.measurements[0]]))
        metadata = ibl.ModelMetadata(ibl.ForwardModelMetadata(x.setup,
            ibl.ParameterInputSchema(x.sample, [ibl.ParameterSpec("thickness", ibl.LayerThickness(0))],
                preprocessing=[ibl.MultiplyScale([.01])]),
            ibl.SpectraSchema([ibl.SpectrumSpec("RBS", 4)])))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trained.ibm"
            package = ibl.export_package(model, torch.from_numpy(inputs[:2]), metadata, path)
            self.assertIsInstance(package, ibl.Package)
            expected = model(torch.ones(1, 1)).detach().numpy()[0]
            np.testing.assert_allclose(ibl.ForwardModel(package).predict(x)["RBS"].counts, expected, rtol=1e-4)
            with self.assertRaises(FileExistsError):
                ibl.export_package(model, torch.from_numpy(inputs[:2]), metadata, path)


if __name__ == "__main__":
    unittest.main()
