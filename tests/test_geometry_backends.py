"""Detector geometry and Python-defined simulators through native generation."""
from dataclasses import replace
from pathlib import Path
import gc
import os
import sys
import tempfile
import unittest
import weakref
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
    sys.path.insert(0, str(Path(sys.argv.pop(1)).resolve()))
ONNX = len(sys.argv) > 1 and sys.argv[1] == "ON"
if len(sys.argv) > 1 and sys.argv[1] in ("ON", "OFF"): sys.argv.pop(1)
import ibeamlab as ibl
from ibeamlab._engine import engine


def experiment(geometry=None):
    return ibl.Experiment(ibl.Sample([ibl.Layer(100, {"Si": 1})]),
        ibl.ExperimentalSetup([ibl.Measurement("RBS", ibl.Beam("H", 2000),
            ibl.Detector(geometry=geometry or ibl.Geometry(0, 15, 165)))]))


class ScalarBackend(ibl.Simulator):
    def __init__(self): self.calls = 0
    def simulate(self, x):
        assert isinstance(x, ibl.Experiment)
        self.calls += 1
        return {m.label: np.full(4, x.sample.layers[0].thickness + m.detector.geometry.scattering_angle)
                for m in x.setup.measurements}


class BackendTests(unittest.TestCase):
    def plan(self): return ibl.GenerationPlan(experiment(), [ibl.vary.layer_thickness(layer=0, bounds=(1, 300))])

    def test_geometry_validation(self):
        for angles in ((0, 15, 165), (30, 40, 120), (0, 180, 0)):
            geometry = ibl.Geometry(*angles)
            geometry.to_core().validate()
        for angles in ((0, 15, 150), (-1, 15, 165), (90, 90, 90), (0, 181, 0),
                       (float("nan"), 0, 180)):
            with self.assertRaises(ibl.ValidationError): ibl.Geometry(*angles)
            native = engine().Geometry()
            native.incidenceAngle, native.exitAngle, native.scatteringAngle = angles
            with self.assertRaises(ValueError): native.validate()

    def test_generation_geometry_parameters_and_python_values(self):
        plan = ibl.GenerationPlan(experiment(), [ibl.vary.exit_angle("RBS", bounds=(0, 90)),
            ibl.vary.scattering_angle("RBS", bounds=(90, 180))])
        backend = ScalarBackend()
        with tempfile.TemporaryDirectory() as tmp:
            progress = []
            result = plan.generate([[15, 165], [20, 160], [30, 150]], backend, Path(tmp) / "data",
                                   batch_size=2, progress=lambda a, b: progress.append((a, b)))
            self.assertEqual(result.accepted, 3)
            self.assertEqual(backend.calls, 3)
            self.assertEqual(progress[-1], (3, 3))
            dataset = ibl.open_dataset(result.path)
            self.assertEqual(dataset.metadata.reference.setup, experiment().setup)
            np.testing.assert_array_equal(dataset.spectra("RBS")[:, 0], [265, 260, 250])
            self.assertIn("ScalarBackend", dataset.metadata.provenance)

    def test_batch_override_is_used(self):
        class BatchBackend(ScalarBackend):
            def __init__(self): self.batches = []
            def simulate(self, x): raise AssertionError("generation must use the batch override")
            def simulate_batch(self, experiments):
                self.batches.append(len(experiments))
                values = np.asarray([x.sample.layers[0].thickness for x in experiments])[:, None] * np.ones((1, 4))
                return [{"RBS": row} for row in values]
        backend = BatchBackend()
        with tempfile.TemporaryDirectory() as tmp:
            result = self.plan().generate([[10], [20], [30], [40], [50]], backend, Path(tmp) / "data", batch_size=2)
            self.assertEqual(backend.batches, [2, 2, 1])
            np.testing.assert_array_equal(ibl.open_dataset(result.path).spectra("RBS")[:, 0], [10, 20, 30, 40, 50])

    def test_failure_policies(self):
        class Failing(ScalarBackend):
            def simulate(self, x):
                if x.sample.layers[0].thickness == 20: raise ValueError("external engine failed")
                return super().simulate(x)
        with tempfile.TemporaryDirectory() as tmp:
            for policy in (ibl.FailurePolicy.RECORD, ibl.FailurePolicy.DISCARD):
                result = self.plan().generate([[10], [20], [30]], Failing(), Path(tmp) / policy.value,
                    failure_policy=policy, batch_size=2)
                self.assertEqual(result.accepted, 2)
                self.assertEqual(result.failed if policy == ibl.FailurePolicy.RECORD else result.discarded, 1)
            with self.assertRaisesRegex(ibl.DatasetError, "external engine failed"):
                self.plan().generate([[10], [20]], Failing(), Path(tmp) / "stop")

    def test_cancellation_and_reset(self):
        class Cancel(ScalarBackend):
            def simulate(self, x):
                result = super().simulate(x)
                if self.calls == 2: self.request_stop()
                return result
        backend = Cancel()
        with tempfile.TemporaryDirectory() as tmp:
            result = self.plan().generate([[10], [20], [30]], backend, Path(tmp) / "cancel", batch_size=3)
            self.assertTrue(result.cancelled)
            self.assertTrue(backend.stopped)
            backend.reset_stop()
            result = self.plan().generate([[10], [20]], backend, Path(tmp) / "complete")
            self.assertEqual(result.accepted, 2)

    def test_invalid_batch_contract(self):
        class Bad(ScalarBackend):
            def simulate_batch(self, experiments): return []
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex((ibl.ValidationError, ibl.DatasetError), "one result"):
                self.plan().generate([[10]], Bad(), Path(tmp) / "bad")

    def test_adapter_does_not_keep_python_owner_alive(self):
        backend = ScalarBackend()
        adapter, reference = backend.to_core(), weakref.ref(backend)
        del backend
        gc.collect()
        self.assertIsNone(reference())
        with self.assertRaises(ibl.SimulationError): adapter.provenance()

    def test_keyboard_interrupt_propagates(self):
        class Interrupt(ScalarBackend):
            def simulate(self, x): raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(KeyboardInterrupt):
                self.plan().generate([[10]], Interrupt(), Path(tmp) / "interrupted")

    def test_reference_geometry(self):
        reference = ROOT / "xnra/Ref_RBS_LiCOFNaAlSiPSTiMnFeCoNiCuH_nopu.xnra"
        self.assertEqual(ibl.SimnraMethod("RBS", reference, "H").reference_geometry(), ibl.Geometry(0, 30, 150))

    @unittest.skipUnless(os.environ.get("IBEAMLAB_TEST_SIMNRA") == "1", "licensed SIMNRA check")
    def test_simnra_geometry_updates(self):
        from cpp.tests.simnra_xnra import load_reference, recalculate_reference
        from ibeamlab import native
        from ibeamlab.sample import experiment_from_core
        reference = ROOT / "xnra/Ref_RBS_LiCOFNaAlSiPSTiMnFeCoNiCuH_nopu.xnra"
        angles = [(0, 20, 160), (10, 10, 160)]
        expected = [recalculate_reference(reference, fast=True, geometry=a) for a in angles]
        raw, _, _ = load_reference(reference, native, expected[0][2], expected[0][4])
        base = experiment_from_core(raw)
        label = base.setup.measurements[0].label
        cases = [replace(base, setup=ibl.ExperimentalSetup([
            replace(base.setup.measurements[0], detector=replace(base.setup.measurements[0].detector,
                geometry=ibl.Geometry(*a)))])) for a in angles]
        with ibl.SimnraSimulator([ibl.SimnraMethod(label, reference, base.setup.measurements[0].beam.particle)],
                                workers=1, fast_calculation=True) as backend:
            results = backend.simulate_batch([cases[0], cases[1], cases[0]])
        for result, index in zip(results, (0, 1, 0)):
            self.assertIsNone(result.failure)
            actual, wanted = result[label].counts, expected[index][0]
            n = max(len(actual), len(wanted))
            np.testing.assert_allclose(np.pad(actual, (0, n-len(actual))),
                                       np.pad(wanted, (0, n-len(wanted))), rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(ONNX, "ONNX extension required")
    def test_model_geometry_contract(self):
        x = experiment(ibl.Geometry(30, 40, 120))
        schema = ibl.SpectraSchema([ibl.SpectrumSpec("RBS", 4)])
        metadata = ibl.ModelMetadata(ibl.ForwardModelMetadata(x.setup,
            ibl.ParameterInputSchema(x.sample, [ibl.ParameterSpec("theta", ibl.ScatteringAngle("RBS"), "degree")]), schema))
        package = ibl.Package.from_onnx(ROOT / "cpp/tests/fixtures/forward.onnx", metadata)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "geometry.ibm"
            package.write(path)
            package = ibl.Package.open(path)
        model = ibl.ForwardModel(package)
        changed = experiment(ibl.Geometry(30, 40, 130))
        np.testing.assert_allclose(model.predict(changed)["RBS"].counts, [130, 260, 390, 520])
        with self.assertRaises(ibl.InferenceError): model.predict(experiment(ibl.Geometry(30, 45, 130)))
        layer_metadata = ibl.ModelMetadata(ibl.ForwardModelMetadata(x.setup, ibl.LayerInputSchema(["Si", "O"]),
            ibl.SpectraSchema([ibl.SpectrumSpec("RBS", 5000)])))
        layer_model = ibl.ForwardModel(ibl.Package.from_onnx(ROOT / "cpp/tests/fixtures/forward_layers.onnx", layer_metadata))
        with self.assertRaisesRegex(ibl.InferenceError, "geometry"):
            layer_model.predict(changed)
        with self.assertRaisesRegex(ibl.InferenceError, "geometry"):
            layer_model.predict_areal_density(np.ones((2, 2), dtype=np.float32), setup=changed.setup)
        inverse = ibl.InverseModel(ibl.Package.from_onnx(ROOT / "cpp/tests/fixtures/inverse.onnx",
            ibl.ModelMetadata(ibl.InverseModelMetadata(x.setup, schema, ibl.EdpSchema(["Si", "O"], 1, True)))))
        with self.assertRaisesRegex(ibl.InferenceError, "geometry"):
            inverse.predict(ibl.InverseInput(changed.setup, {"RBS": [1, 2, 3, 4]}))


if __name__ == "__main__": unittest.main()
