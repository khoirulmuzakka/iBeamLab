"""Focused metadata/inference boundary checks; no training or production data."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if len(sys.argv) > 1:
    sys.path.insert(0, str(Path(sys.argv.pop(1)).resolve()))
import ibeamlab as ibl
from ibeamlab._engine import engine

FIXTURES = ROOT / "cpp/tests/fixtures"


def experiment():
    return ibl.Experiment(ibl.Sample([ibl.Layer(100, {"Si": .5, "O": .5})]),
        ibl.ExperimentalSetup([ibl.Measurement("RBS", ibl.Beam("H", 2000),
            acquisition=ibl.Acquisition(1000, 10, 8))]))


def forward(specs=None):
    x = experiment()
    return ibl.ForwardModelMetadata(x.setup,
        ibl.ParameterInputSchema(x.sample, [ibl.ParameterSpec("thickness", ibl.LayerThickness(0))]),
        ibl.SpectraSchema(specs or [ibl.SpectrumSpec("RBS", 4)]))


class StagesTests(unittest.TestCase):
    def test_reference_and_measurement_modes(self):
        f = forward()
        f.output_transform = [ibl.SpectrumTransform("RBS", ibl.TransformPipeline(
            [ibl.MultiplyScale([.1])], direction=ibl.TransformDirection.INVERSE))]
        f.output_preparation = [ibl.SpectrumProcessing("RBS", [ibl.ExposureScale(), ibl.Pileup(4e-7)])]
        package = ibl.Package.from_onnx(FIXTURES / "forward.onnx", ibl.ModelMetadata(f))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "forward.ibm"
            package.write(path)
            self.assertEqual(path.read_bytes()[:8], b"IBMODEL1")
            model = ibl.load_model(path)
            self.assertEqual(model.metadata.model.output_transform[0].transform.direction,
                             ibl.TransformDirection.INVERSE)
            x = experiment()
            requested = replace(x, setup=replace(x.setup, measurements=[replace(x.setup.measurements[0],
                acquisition=ibl.Acquisition(2000, 20, 10))]))
            reference = model.predict_reference(requested)["RBS"].counts
            np.testing.assert_allclose(reference, [1000, 2000, 3000, 4000])
            expected = ibl.apply_pileup(reference * 2, real_time=20, live_time=10, tau=4e-7)
            np.testing.assert_allclose(model.predict(requested)["RBS"].counts, expected)
            for result in model.predict_reference_batch([requested, x]):
                np.testing.assert_allclose(result["RBS"].counts, reference)

    def test_graph_and_runtime_placements_execute_once(self):
        import onnx
        from onnx import helper as h, numpy_helper as n, TensorProto as t
        with tempfile.TemporaryDirectory() as tmp:
            for input_graph in (False, True):
                for output_graph in (False, True):
                    nodes = []
                    source = "inputs"
                    if input_graph:
                        nodes.append(h.make_node("Mul", [source, "two"], ["scaled_input"]))
                        source = "scaled_input"
                    nodes.append(h.make_node("MatMul", [source, "weights"], ["raw"]))
                    nodes.append(h.make_node("Mul" if output_graph else "Identity",
                        ["raw", "ten"] if output_graph else ["raw"], ["outputs"]))
                    initializers = [n.from_array(np.array([[1, 2, 3, 4]], np.float32), "weights")]
                    if input_graph:
                        initializers.append(n.from_array(np.array(2, np.float32), "two"))
                    if output_graph:
                        initializers.append(n.from_array(np.array(10, np.float32), "ten"))
                    graph = h.make_graph(nodes, "stages", [h.make_tensor_value_info("inputs", t.FLOAT, [None, 1])],
                        [h.make_tensor_value_info("outputs", t.FLOAT, [None, 4])],
                        initializers)
                    path = Path(tmp) / "stages.onnx"
                    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 13)], ir_version=8), path)
                    f = forward()
                    f.input_transform = ibl.TransformPipeline([ibl.MultiplyScale([2])], location=(
                        ibl.TransformLocation.GRAPH if input_graph else ibl.TransformLocation.RUNTIME))
                    f.output_transform = [ibl.SpectrumTransform("RBS", ibl.TransformPipeline(
                        [ibl.MultiplyScale([.1])], direction=ibl.TransformDirection.INVERSE, location=(
                            ibl.TransformLocation.GRAPH if output_graph else ibl.TransformLocation.RUNTIME)))]
                    model = ibl.ForwardModel(ibl.Package.from_onnx(path, ibl.ModelMetadata(f)))
                    np.testing.assert_allclose(model.predict_reference(experiment())["RBS"].counts,
                                               [2000, 4000, 6000, 8000])

    def test_independent_inverse_transforms_and_uncertainty(self):
        x = experiment()
        setup = replace(x.setup, measurements=[x.setup.measurements[0],
            replace(x.setup.measurements[0], label="NRA")])
        metadata = ibl.ModelMetadata(ibl.InverseModelMetadata(setup,
            ibl.SpectraSchema([ibl.SpectrumSpec("RBS", 4), ibl.SpectrumSpec("NRA", 3)]),
            ibl.EdpSchema(["Si", "O"], 1, True),
            input_preparation=[ibl.SpectrumProcessing(label, [ibl.ExposureScale()]) for label in ("RBS", "NRA")],
            input_transform=[ibl.SpectrumTransform("RBS", ibl.TransformPipeline([ibl.StandardScale([0], [2])])),
                             ibl.SpectrumTransform("NRA", ibl.TransformPipeline([ibl.Log1p()]))],
            output_transform=ibl.TransformPipeline([ibl.StandardScale([1, 2], [2, 3])], direction=ibl.TransformDirection.INVERSE)))
        package = ibl.Package.from_onnx(FIXTURES / "inverse_multi.onnx", metadata)
        model = ibl.InverseModel(package)
        spectra = {"RBS": np.array([1, 2, 3, 4.]), "NRA": np.array([1, 2, 3.])}
        ready = np.concatenate([spectra["RBS"] / 2, np.log1p(spectra["NRA"])]).astype(np.float32)[None]
        raw = engine().run_graph(package.to_core(), ready)[0]
        result = model.predict_reference(spectra)
        np.testing.assert_allclose(result.values[0], raw[:2] * [2, 3] + [1, 2])
        np.testing.assert_allclose(result.presence_probability[0], raw[2:4])
        np.testing.assert_allclose(result.posterior_std[0], raw[4:6] * [2, 3])
        requested = replace(setup, measurements=[replace(m, acquisition=ibl.Acquisition(2000, 10, 8)) for m in setup.measurements])
        physical = model.predict(ibl.InverseInput(requested, {k: v * 2 for k, v in spectra.items()}))
        np.testing.assert_allclose(physical.values, result.values)
        for row in model.predict_reference_batch([spectra, spectra]):
            np.testing.assert_allclose(row.values, result.values)

    def test_reject_ambiguous_or_noninvertible_contracts(self):
        x = experiment()
        for steps in ([ibl.Log1p()], [ibl.MinMaxScale([0], [1], clip=True)]):
            meta = ibl.ModelMetadata(ibl.InverseModelMetadata(x.setup,
                ibl.SpectraSchema([ibl.SpectrumSpec("RBS", 4)]), ibl.EdpSchema(["Si", "O"], 1, True),
                output_transform=ibl.TransformPipeline(steps)))
            with self.assertRaises(ibl.ValidationError):
                meta.validate()
        f = forward()
        f.input_transform = ibl.TransformPipeline([ibl.MultiplyScale([0])], direction=ibl.TransformDirection.INVERSE)
        with self.assertRaisesRegex(ibl.ValidationError, "zero scaling"):
            ibl.ModelMetadata(f).validate()
        f = forward([ibl.SpectrumSpec("RBS", 4, ibl.SpectrumState.PILED_UP)])
        f.output_preparation = [ibl.SpectrumProcessing("RBS", [ibl.Pileup(4e-7)])]
        with self.assertRaisesRegex(ibl.ValidationError, "bare"):
            ibl.ModelMetadata(f).validate()

    def test_direct_population_reference_output_reuse(self):
        x = experiment()
        f = ibl.ForwardModelMetadata(x.setup, ibl.LayerInputSchema(["Si", "O"]),
            ibl.SpectraSchema([ibl.SpectrumSpec("RBS", 5000)]),
            input_transform=ibl.TransformPipeline([ibl.MultiplyScale([.01, .02])]),
            output_transform=[ibl.SpectrumTransform("RBS", ibl.TransformPipeline([ibl.MultiplyScale([2])]))],
            output_preparation=[ibl.SpectrumProcessing("RBS", [ibl.Pileup(1e-12)])])
        model = ibl.ForwardModel(ibl.Package.from_onnx(FIXTURES / "forward_layers.onnx", ibl.ModelMetadata(f)))
        ad = np.ones((4, 3, 2), np.float32) * 50
        saved = ad.copy()
        out = np.empty((4, 5000))
        actual = model.predict_reference_areal_density_batch(ad, out=out)["RBS"]
        self.assertTrue(np.shares_memory(actual, out))
        expected = model.predict_reference_areal_density(ad[0])["RBS"].counts
        np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape))
        self.assertEqual(model.predict_areal_density_batch(ad)["RBS"].shape, (4, 9999))
        np.testing.assert_array_equal(ad, saved)


if __name__ == "__main__":
    unittest.main()
