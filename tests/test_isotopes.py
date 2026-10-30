"""Scientific isotope defaults, package assumptions and inference compatibility."""
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


def experiment(isotopes=()):
    return ibl.Experiment(ibl.Sample([ibl.Layer(100, [ibl.Species("Si", .5, isotopes),
                                                      ibl.Species("O", .5)])]),
        ibl.ExperimentalSetup([ibl.Measurement("RBS", ibl.Beam("H", 2000))]))


def package(compositions=(), sequence=True):
    x = experiment()
    schema = (ibl.LayerInputSchema(["Si", "O"], compositions) if sequence else
              ibl.ParameterInputSchema(x.sample, [ibl.ParameterSpec("thickness", ibl.LayerThickness(0))]))
    metadata = ibl.ModelMetadata(ibl.ForwardModelMetadata(x.setup, schema,
        ibl.SpectraSchema([ibl.SpectrumSpec("RBS", 5000 if sequence else 4)])))
    return ibl.Package.from_onnx(ROOT / "cpp/tests/fixtures" /
                               ("forward_layers.onnx" if sequence else "forward.onnx"), metadata)


class IsotopeTests(unittest.TestCase):
    def test_natural_and_explicit_sample_validation(self):
        natural = ibl.natural_isotopes("Si")
        self.assertEqual({i.mass_number for i in natural}, {28, 29, 30})
        self.assertAlmostEqual(sum(i.fraction for i in natural), 1)
        self.assertTrue({28, 29, 30}.issubset({i.massNumber for i in ibl.isotope_data("Si")}))
        experiment().to_core().validate()
        experiment(natural).to_core().validate()
        experiment((ibl.Isotope(28),)).to_core().validate()
        for values in ((ibl.Isotope(999),), (ibl.Isotope(28, .5), ibl.Isotope(28, .5))):
            with self.assertRaises(ValueError): experiment(values).to_core()
        with self.assertRaises(ibl.ValidationError): ibl.Isotope(0)
        self.assertEqual(ibl.natural_isotopes("Tc"), ())
        with self.assertRaises(ValueError): ibl.Sample([ibl.Layer(1, {"Tc": 1})]).to_core()

    def test_natural_inference_and_enrichment_rejection(self):
        for sequence in (True, False):
            model = ibl.ForwardModel(package(sequence=sequence))
            expected = model.predict(experiment())["RBS"].counts
            values = ibl.natural_isotopes("Si")
            for composition in (values, values[::-1], (*values, ibl.Isotope(31, 0.0))):
                actual = model.predict(experiment(composition))["RBS"].counts
                np.testing.assert_array_equal(actual, expected)
            with self.assertRaises(ibl.InferenceError):
                model.predict(experiment((ibl.Isotope(28),)))

    def test_enriched_model_and_package_round_trip(self):
        p = package([ibl.ElementComposition("Si", (ibl.Isotope(28),))])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ibm"
            p.write(path)
            reopened = ibl.Package.open(path)
        schema = reopened.metadata.model.input_preparation
        self.assertEqual([c.element for c in schema.isotope_compositions], ["Si", "O"])
        self.assertEqual(schema.isotope_compositions[0].isotopes, (ibl.Isotope(28),))
        self.assertTrue(schema.isotope_compositions[1].isotopes)
        model = ibl.ForwardModel(reopened)
        model.predict(experiment((ibl.Isotope(28),)))
        with self.assertRaisesRegex(ibl.InferenceError, "Si"):
            model.predict(experiment())
        with self.assertRaisesRegex(ibl.InferenceError, "Si"):
            model.predict(experiment(ibl.natural_isotopes("Si")))

    def test_invalid_model_assumptions(self):
        for compositions in ([ibl.ElementComposition("C")],
                [ibl.ElementComposition("Si"), ibl.ElementComposition("Si")],
                [ibl.ElementComposition("Si", (ibl.Isotope(999),))]):
            with self.assertRaises((ValueError, ibl.ModelPackageError)):
                package(compositions)

    @unittest.skipUnless(os.environ.get("IBEAMLAB_TEST_SIMNRA") == "1", "licensed SIMNRA check")
    def test_simnra_natural_enriched_and_restored(self):
        from cpp.tests.simnra_xnra import load_reference
        from ibeamlab import native
        from ibeamlab.sample import experiment_from_core
        reference = ROOT / "xnra/Ref_RBS_LiCOFNaAlSiPSTiMnFeCoNiCuH_nopu.xnra"
        raw, _, _ = load_reference(reference, native)
        x = experiment_from_core(raw)
        x = replace(x, setup=ibl.ExperimentalSetup([replace(x.setup.measurements[0], label="RBS")]))
        explicit = replace(x, sample=ibl.Sample([ibl.Layer(layer.thickness, [
            replace(species, isotopes=ibl.natural_isotopes(species.element)[::-1])
            for species in layer.species]) for layer in x.sample.layers]))
        enriched = replace(x, sample=ibl.Sample([ibl.Layer(layer.thickness, [
            replace(species, isotopes=(ibl.Isotope(28),)) if species.element == "Si" else species
            for species in layer.species]) for layer in x.sample.layers]))
        with ibl.SimnraSimulator([ibl.SimnraMethod("RBS", reference, x.setup.measurements[0].beam.particle)],
                                workers=1, fast_calculation=True) as simulator:
            def predict(value):
                result = simulator.simulate(value)
                self.assertIsNone(result.failure)
                return result["RBS"].counts
            baseline, specified = predict(x), predict(explicit)
            altered, restored = predict(enriched), predict(x)
        def padded(a, b):
            n = max(len(a), len(b))
            return np.pad(a, (0, n-len(a))), np.pad(b, (0, n-len(b)))
        np.testing.assert_allclose(*padded(specified, baseline), rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(*padded(restored, baseline), rtol=1e-10, atol=1e-8)
        a, b = padded(altered, baseline)
        self.assertGreater(np.max(np.abs(a-b)), 1e-6)

    def test_export_rejects_conflicting_training_mixtures(self):
        from dataclasses import asdict
        from examples.ibeamnet.data import model_isotope_compositions
        def source(x): return {"reference": asdict(x)}
        manifest = {"elements": ["Si", "O"], "sources": [source(experiment()),
                     source(experiment(ibl.natural_isotopes("Si")[::-1]))]}
        compositions = model_isotope_compositions(manifest)
        self.assertEqual([c.element for c in compositions], ["Si", "O"])
        manifest["sources"].append(source(experiment((ibl.Isotope(28),))))
        with self.assertRaisesRegex(ValueError, "Si"):
            model_isotope_compositions(manifest)


if __name__ == "__main__": unittest.main()
