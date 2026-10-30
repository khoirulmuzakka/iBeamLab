"""Native CPU/CUDA provider parity. Set IBEAMLAB_TEST_CUDA=1 to require GPU execution."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
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
    inputs = ibl.LayerInputSchema(["Si", "O"]) if sequence else ibl.ParameterInputSchema(
        x.sample, [ibl.ParameterSpec("thickness", ibl.LayerThickness(0))])
    metadata = ibl.ModelMetadata(ibl.ForwardModelMetadata(x.setup, inputs,
        ibl.SpectraSchema([ibl.SpectrumSpec("RBS", 5000 if sequence else 4)]),
        input_transform=ibl.TransformPipeline([ibl.MultiplyScale([.01])] if sequence else [])))
    return ibl.Package.from_onnx(FIXTURES / ("forward_layers.onnx" if sequence else "forward.onnx"), metadata)


def inverse_package():
    return ibl.Package.from_onnx(FIXTURES / "inverse.onnx", ibl.ModelMetadata(ibl.InverseModelMetadata(reference().setup, ibl.SpectraSchema([ibl.SpectrumSpec('RBS', 4)]), ibl.EdpSchema(['Si', 'O'], 1, True))))


class ProviderTests(unittest.TestCase):
    def test_population_batch_and_reused_output(self):
        providers = ["cpu", "cuda"] if GPU else ["cpu"]
        rng = np.random.default_rng(42)
        for provider in providers:
            for preprocessing in (True, False):
                package = forward_package(sequence=True)
                if not preprocessing:
                    metadata = package.metadata
                    metadata.model.input_transform.steps = []
                    package.metadata = metadata
                model = ibl.ForwardModel(package, options=ibl.InferenceOptions(execution_provider=provider))
                for batch, layers in ((1, 1), (7, 8), (3, 2), (16, 17), (7, 8), (0, 2)):
                    ad = rng.uniform(0, 100, (batch, layers, 2)).astype(np.float32)
                    if layers > 2:
                        ad[:, 1] = 0
                        ad[-1] = 0
                    saved = ad.copy()
                    out = np.empty((batch, 5000), dtype=np.float64)
                    actual = model.predict_areal_density_batch(ad, out=out)["RBS"]
                    self.assertEqual(actual.shape, (batch, 5000))
                    if batch:
                        self.assertTrue(np.shares_memory(actual, out))
                    expected = np.asarray([model.predict_areal_density(row)["RBS"].counts for row in ad])
                    if batch:
                        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
                        result = model.predict_areal_density_batch(ad)["RBS"]
                        model.predict_areal_density_batch(ad * 2, out=out)
                        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
                    np.testing.assert_array_equal(ad, saved)
                for bad in (np.ones((3, 2)), np.ones((2, 0, 2)), np.ones((2, 3, 3)),
                            np.full((2, 3, 2), -1), np.full((2, 3, 2), np.nan)):
                    with self.assertRaises(ibl.InferenceError):
                        model.predict_areal_density_batch(bad)
                ad = np.ones((2, 3, 2), np.float32)
                for bad_out in (np.empty((2, 5000), np.float32), np.empty((1, 5000)),
                                np.empty((5000, 2)).T):
                    with self.assertRaises(ibl.InferenceError):
                        model.predict_areal_density_batch(ad, out=bad_out)
                readonly = np.empty((2, 5000))
                readonly.flags.writeable = False
                with self.assertRaises(ibl.InferenceError):
                    model.predict_areal_density_batch(ad, out=readonly)
                storage = np.zeros((2, 5000), np.float64)
                overlapping = storage.view(np.float32).reshape(-1)[:12].reshape(2, 3, 2)
                with self.assertRaisesRegex(ibl.InferenceError, "overlap"):
                    model.predict_areal_density_batch(overlapping, out=storage)

    def test_experiment_and_inverse_batches(self):
        for provider in (["cpu", "cuda"] if GPU else ["cpu"]):
            options = ibl.InferenceOptions(execution_provider=provider)
            for sequence in (True, False):
                model = ibl.ForwardModel(forward_package(sequence=sequence), options=options)
                cases = [replace(reference(), sample=ibl.Sample([
                    ibl.Layer(100 + n, {"Si": .5, "O": .5}) for _ in range(n if sequence else 1)]))
                    for n in (1, 3, 8)]
                actual = model.predict_batch(cases)
                for x, result in zip(cases, actual):
                    np.testing.assert_allclose(result["RBS"].counts, model.predict(x)["RBS"].counts, rtol=1e-5)
                self.assertEqual(model.predict_batch([]), [])
            inverse = ibl.InverseModel(inverse_package(), options=options)
            inputs = [ibl.InverseInput(reference().setup, {"RBS": [v, 2*v, 3*v, 4*v]}) for v in (1, 2, 4)]
            for x, result in zip(inputs, inverse.predict_batch(inputs)):
                expected = inverse.predict(x)
                np.testing.assert_allclose(result.values, expected.values)
                np.testing.assert_allclose(result.presence_probability, expected.presence_probability)
                np.testing.assert_allclose(result.posterior_std, expected.posterior_std)
            self.assertEqual(inverse.predict_batch([]), [])

    def test_independent_detector_processing_in_population(self):
        reference_setup = reference().setup
        second = replace(reference_setup.measurements[0], label="second")
        setup = ibl.ExperimentalSetup([reference_setup.measurements[0], second])
        metadata = ibl.ModelMetadata(ibl.ForwardModelMetadata(setup, ibl.LayerInputSchema(['Si', 'O']), ibl.SpectraSchema([ibl.SpectrumSpec('RBS', 2500), ibl.SpectrumSpec('second', 2500)]), input_transform=ibl.TransformPipeline([ibl.MultiplyScale([0.01, 0.02])]), output_preparation=[ibl.SpectrumProcessing('RBS', [ibl.ExposureScale()]), ibl.SpectrumProcessing('second', [ibl.Rebin(1200, False), ibl.Pileup(1e-12)])], output_transform=[ibl.SpectrumTransform('RBS', ibl.TransformPipeline([ibl.MultiplyScale([2])])), ibl.SpectrumTransform('second', ibl.TransformPipeline([ibl.MultiplyScale([3])]))]))
        package = ibl.Package.from_onnx(FIXTURES / "forward_layers.onnx", metadata)
        for provider in (["cpu", "cuda"] if GPU else ["cpu"]):
            model = ibl.ForwardModel(package, options=ibl.InferenceOptions(execution_provider=provider))
            ad = np.full((3, 4, 2), 50, np.float32)
            changed = replace(setup, measurements=[replace(setup.measurements[0],
                acquisition=replace(setup.measurements[0].acquisition, particles_sr=2000)),
                replace(second, acquisition=replace(second.acquisition, real_time=20, live_time=10))])
            out = np.empty((3, 2500 + 2399))
            actual = model.predict_areal_density_batch(ad, setup=changed, out=out)
            self.assertEqual(actual["second"].shape, (3, 2399))
            expected = model.predict_areal_density(ad[0], setup=changed)
            bare = ibl.ForwardModel(package, options=ibl.InferenceOptions(execution_provider=provider, apply_pileup=False))
            bare_population = bare.predict_areal_density_batch(ad, setup=changed)
            self.assertEqual(bare_population["second"].shape, (3, 1200))
            np.testing.assert_allclose(ibl.apply_pileup(bare_population["second"][0],
                real_time=20, live_time=10, tau=1e-12), actual["second"][0], rtol=1e-5)
            for label in ("RBS", "second"):
                self.assertTrue(np.shares_memory(actual[label], out))
                np.testing.assert_allclose(actual[label], np.broadcast_to(expected[label].counts,
                                           actual[label].shape), rtol=1e-5)
            experiments = [ibl.Experiment(ibl.Sample([ibl.Layer(100, {"Si": .5, "O": .5})] * n),
                                          changed if n % 2 else setup) for n in (1, 2, 3)]
            for x, result in zip(experiments, model.predict_batch(experiments)):
                expected = model.predict(x)
                for label in ("RBS", "second"):
                    np.testing.assert_allclose(result[label].counts, expected[label].counts, rtol=1e-5)

    def test_batch_executes_population_together(self):
        # A batch-dependent graph detects any regression to repeated single calls.
        import onnx
        from onnx import helper as h, numpy_helper as n, TensorProto as t
        graph = h.make_graph([
            h.make_node("MatMul", ["input", "weights"], ["rows"]),
            h.make_node("ReduceSum", ["rows", "axis"], ["total"], keepdims=1),
            h.make_node("Add", ["rows", "total"], ["output"]),
        ], "population", [h.make_tensor_value_info("input", t.FLOAT, [None, 1])],
           [h.make_tensor_value_info("output", t.FLOAT, [None, 4])],
           [n.from_array(np.ones((1, 4), np.float32), "weights"),
            n.from_array(np.array([0], np.int64), "axis")])
        graph_model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)], ir_version=8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "population.onnx"
            onnx.save(graph_model, path)
            metadata = forward_package().metadata
            metadata = replace(metadata, input_name="input", output_name="output")
            package = ibl.Package.from_onnx(path, metadata)
            model = ibl.ForwardModel(package)
            cases = [replace(reference(), sample=ibl.Sample([ibl.Layer(v, {"Si": .5, "O": .5})]))
                     for v in (100, 200, 300)]
            result = model.predict_batch(cases)
            for expected, row in zip((700, 800, 900), result):
                np.testing.assert_allclose(row["RBS"].counts, expected)
            graph_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
            graph_model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1
            onnx.save(graph_model, path)
            fixed = ibl.ForwardModel(ibl.Package.from_onnx(path, metadata))
            with self.assertRaisesRegex(ibl.InferenceError, "batch shape"):
                fixed.predict_batch(cases)

    def test_parallel_load_and_population_workers(self):
        workers = 16
        barrier = Barrier(workers)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shared.ibm"
            forward_package(sequence=True).write(path)
            def fit_worker(index):
                barrier.wait(timeout=30)
                provider = "cuda" if GPU and index % 2 else "cpu"
                model = ibl.load_model(path, options=ibl.InferenceOptions(execution_provider=provider))
                ad = np.full((8, 3, 2), index + 1, np.float32)
                out = np.empty((8, 5000))
                for _ in range(5):
                    result = model.predict_areal_density_batch(ad, out=out)["RBS"]
                expected = model.predict_areal_density(ad[0])["RBS"].counts
                np.testing.assert_allclose(result, np.broadcast_to(expected, result.shape), rtol=1e-5)
                return provider
            with ThreadPoolExecutor(max_workers=workers) as pool:
                providers = list(pool.map(fit_worker, range(workers)))
            self.assertIn("cpu", providers)
            if GPU:
                self.assertIn("cuda", providers)

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
        result = inverse.predict_reference({"RBS": [1, 2, 3, 4]})
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
                        ibl.InferenceOptions(allow_tf32=1), ibl.InferenceOptions(apply_pileup=1),
                        ibl.InferenceOptions(device_id=2**100)):
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
                actual = gpu.predict_reference({"RBS": [1, 2, 3, 4]})
                expected = cpu.predict_reference({"RBS": [1, 2, 3, 4]})
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
