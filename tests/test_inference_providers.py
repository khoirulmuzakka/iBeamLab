"""Native CPU/CUDA provider parity. Set IBEAMLAB_TEST_CUDA=1 to require GPU execution."""
from dataclasses import replace
from pathlib import Path
import os
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
    sys.path.insert(0, str(Path(sys.argv.pop(1)).resolve()))
import ibeamlab as ibl
from ibeamlab._engine import engine

GPU = os.environ.get("IBEAMLAB_TEST_CUDA") == "1"
FIXTURES = ROOT / "cpp/tests/fixtures"


def reference():
    return ibl.Experiment(ibl.Sample([ibl.Layer(100, {"Si": .5, "O": .5})]),
        ibl.ExperimentalSetup([ibl.Measurement("RBS", ibl.Beam("H", 2000),
                                              acquisition=ibl.Acquisition(1000, 10, 8))]))


def forward_package(*, sequence=False):
    x = reference()
    inputs = ibl.LayerInputSchema(["Si", "O"], [ibl.MultiplyScale([.01])]) if sequence else ibl.ParameterInputSchema(
        x.sample, [ibl.ParameterSpec("thickness", ibl.LayerThickness(0))])
    metadata = ibl.ModelMetadata(ibl.ForwardModelMetadata(x.setup, inputs,
        ibl.SpectraSchema([ibl.SpectrumSpec("RBS", 5000 if sequence else 4)])))
    return ibl.Package.from_onnx(FIXTURES / ("forward_layers.onnx" if sequence else "forward.onnx"), metadata)


def inverse_package():
    return ibl.Package.from_onnx(FIXTURES / "inverse.onnx", ibl.ModelMetadata(ibl.InverseModelMetadata(
        reference().setup, ibl.SpectraSchema([ibl.SpectrumSpec("RBS", 4)]), ["Si", "O"], 1, True)))


class ProviderTests(unittest.TestCase):
    def test_cpu_options_and_package_independence(self):
        options = ibl.InferenceOptions(intra_op_threads=2, inter_op_threads=1, enable_graph_optimizations=False)
        package = forward_package()
        self.assertIn("cpu", ibl.available_execution_providers())
        model = ibl.ForwardModel(package, options=options)
        self.assertEqual(model.options, options)
        np.testing.assert_allclose(model.predict(reference())["RBS"].counts, [100, 200, 300, 400])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forward.ibm"
            package.write(path)
            loaded = ibl.load_model(path, options=options)
            self.assertEqual(loaded.options, options)
            self.assertFalse(hasattr(loaded.metadata, "execution_provider"))
        inverse = ibl.InverseModel(inverse_package(), options=options)
        result = inverse.predict_prepared({"RBS": [1, 2, 3, 4]})
        np.testing.assert_allclose(result.values, [[10, 20]])
        self.assertEqual(inverse.options, options)
        raw = options.to_core()
        snapshot = ibl.ForwardModel(package, options=raw)
        raw.intra_op_threads = 8
        self.assertEqual(snapshot.options.intra_op_threads, 2)

    def test_invalid_options(self):
        for options in (ibl.InferenceOptions(execution_provider="invalid"), ibl.InferenceOptions(device_id=-1),
                        ibl.InferenceOptions(intra_op_threads=0), ibl.InferenceOptions(inter_op_threads=0),
                        ibl.InferenceOptions(device_id=True), ibl.InferenceOptions(enable_graph_optimizations=1),
                        ibl.InferenceOptions(allow_tf32=1), ibl.InferenceOptions(device_id=2**100)):
            with self.assertRaises(ibl.ValidationError):
                options.to_core()

    def test_cuda_request_cannot_silently_use_cpu(self):
        if "cuda" in ibl.available_execution_providers():
            self.skipTest("GPU-enabled build; CUDA failures are checked in GPU tests")
        options = ibl.InferenceOptions(execution_provider="cuda")
        for cls, package in ((ibl.ForwardModel, forward_package()), (ibl.InverseModel, inverse_package())):
            with self.assertRaisesRegex(ibl.InferenceError, "CUDA requested"):
                cls(package, options=options)

    @unittest.skipUnless(GPU, "set IBEAMLAB_TEST_CUDA=1 on a CUDA-capable machine")
    def test_cuda_forward_inverse_and_dynamic_layers(self):
        self.assertIn("cuda", ibl.available_execution_providers())
        cuda = ibl.InferenceOptions(execution_provider="cuda")
        for cls, package in ((ibl.ForwardModel, forward_package()), (ibl.InverseModel, inverse_package())):
            gpu, cpu = cls(package, options=cuda), cls(package)
            self.assertEqual(gpu.options.execution_provider, "cuda")
            if cls is ibl.ForwardModel:
                np.testing.assert_allclose(gpu.predict(reference())["RBS"].counts, cpu.predict(reference())["RBS"].counts)
            else:
                actual = gpu.predict_prepared({"RBS": [1, 2, 3, 4]})
                expected = cpu.predict_prepared({"RBS": [1, 2, 3, 4]})
                np.testing.assert_allclose(actual.values, expected.values)
                np.testing.assert_allclose(actual.posterior_std, expected.posterior_std)
        package = forward_package(sequence=True)
        for optimization in (False, True):
            options = replace(cuda, enable_graph_optimizations=optimization)
            gpu, cpu = ibl.ForwardModel(package, options=options), ibl.ForwardModel(package)
            for count in (1, 2, 8, 32, 2):
                ad = np.full((count, 2), 50., np.float32)
                if count > 2:
                    ad[1] = 0
                saved = ad.copy()
                np.testing.assert_allclose(gpu.predict_areal_density(ad)["RBS"].counts,
                                           cpu.predict_areal_density(ad)["RBS"].counts, rtol=1e-5)
                np.testing.assert_array_equal(ad, saved)
                x = ibl.Experiment(ibl.Sample([ibl.Layer(100, {"O": .5, "Si": .5}) for _ in range(count)]), reference().setup)
                np.testing.assert_allclose(gpu.predict(x)["RBS"].counts, cpu.predict(x)["RBS"].counts, rtol=1e-5)
                for batch in (1, 3):
                    ready = np.tile(ad[None], (batch, 1, 1)) * .01
                    np.testing.assert_allclose(engine().run_graph(package.to_core(), ready, options.to_core()),
                                               engine().run_graph(package.to_core(), ready), rtol=1e-5)
        with self.assertRaisesRegex(ibl.InferenceError, "CUDA"):
            ibl.ForwardModel(package, options=replace(cuda, device_id=100000))
        fast = ibl.ForwardModel(forward_package(), options=replace(cuda, allow_tf32=True))
        self.assertTrue(fast.options.allow_tf32)
        np.testing.assert_allclose(fast.predict(reference())["RBS"].counts, [100, 200, 300, 400])


if __name__ == "__main__":
    unittest.main()
