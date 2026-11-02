"""Backend fits, routing, parser metadata and independent-worker execution."""
import sys
from pathlib import Path
import unittest
import tempfile
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if len(sys.argv)>1 and Path(sys.argv[1]).is_dir():
    sys.path.insert(0, str(Path(sys.argv.pop(1)).resolve()))
import ibeamlab as ibl
from ibeamlab._engine import engine


class Inventory(ibl.Simulator):
    def __init__(self):
        self.batches = []

    def simulate(self, experiment):
        value = sum(l.thickness*l.species[0].concentration for l in experiment.sample.layers)
        return {m.label: [value, value] for m in experiment.setup.measurements}

    def simulate_batch(self, experiments):
        self.batches.append(len(experiments))
        return [self.simulate(e) for e in experiments]


class AnalysisTests(unittest.TestCase):
    def problem(self, predictor=None, labels=("A",)):
        setup = ibl.ExperimentalSetup([ibl.Measurement(name, ibl.Beam("H", 2000)) for name in labels])
        sample = ibl.Sample([ibl.Layer(100, {"Si": .5, "O": .5})])
        data = ibl.MeasuredData(setup, [ibl.MeasuredSpectrum(name, [50, 50]) for name in labels])
        return ibl.FitProblem(data, sample, [ibl.vary.layer_thickness(layer=0, bounds=(50, 200))], predictor or Inventory())

    def test_composite_batch_and_reduction(self):
        a, b = Inventory(), Inventory()
        problem = self.problem(ibl.CompositePredictor({"A": a, "B": b}), ("B", "A"))
        objective = ibl.FitObjective(problem)
        values = objective.evaluate_batch([[100], [120], [80]])
        self.assertEqual(values[0], 0)
        self.assertGreater(values[1], 0)
        objective.evaluate_batch([[110]])
        self.assertEqual(a.batches, [1, 3, 1])
        self.assertEqual(b.batches, [1, 3, 1])

    def test_profile_fit_and_parameter_reporting(self):
        p = self.problem()
        p.parameters = []
        p.profile = ibl.ProfileModel((ibl.ProfileLayer(p.initial_sample.layers[0],
                         (ibl.polynomial(1, -1), ibl.polynomial(0, 1)), 2),))
        p.profile_parameters = (ibl.ProfileParameter(0, 0, 1, (-1, 0), "slope"),)
        o = ibl.FitObjective(p)
        self.assertAlmostEqual(o.best_result().cost, 0)
        self.assertEqual(o.best_result().parameters["slope"], -1)
        self.assertEqual(len(o.decode([-.5]).sample.layers), 2)
        self.assertGreater(o.evaluate_batch([[-.5]])[0], 0)
        self.assertTrue(np.isinf(o.evaluate_batch([[-2]])[0]))

    def test_losses_and_ratio_links(self):
        for loss, expected in [("data_chi_square", 200/51), ("neyman", 4), ("pearson", 200/60), ("mse", 200)]:
            p = self.problem()
            p.loss = loss
            o = ibl.FitObjective(p)
            self.assertAlmostEqual(o.evaluate_batch([[120]])[0], expected)
        p = self.problem(labels=("A", "B"))
        p.parameters = [ibl.LinkedParameter(ibl.vary.calibration_linear("A", bounds=(.5, 2)),
                                            ((ibl.CalibrationLinear("B"), 1),))]
        o = ibl.FitObjective(p)
        values = [m.detector.calibration_linear for m in o.decode([1.5]).setup.measurements]
        self.assertEqual(values, [1.5, 1.5])

    def test_parallel_tasks_and_initialization_dependencies(self):
        executor = ibl.AnalysisExecutor(2, lambda worker: Inventory())
        tasks = [ibl.AnalysisTask("first", self.problem(), {"max_evaluations": 1}),
                 ibl.AnalysisTask("second", self.problem(), {"max_evaluations": 1}, ("first",))]
        out = executor.run(tasks)
        self.assertEqual([v.id for v in out], ["first", "second"])
        self.assertTrue(all(not v.error and v.result.cost == 0 for v in out))
        if engine().arrde_available():
            staged = executor.run([ibl.AnalysisTask("staged", self.problem(), {"stages": [
                        {"max_evaluations": 1}, {"max_evaluations": 8, "population_size": 4}]})])
            self.assertEqual(staged[0].error, "")
            self.assertEqual(staged[0].result.cost, 0)
        with self.assertRaises(ValueError):
            ibl.AnalysisExecutor(2, lambda worker: executor.providers[0])
        with self.assertRaises(ValueError):
            executor.run([ibl.AnalysisTask("x", self.problem(), initial_from=("x",))])

    def test_parser_and_result_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/"spectrum.txt"
            path.write_text("Real: 2000\nLive: 1500\nBEGIN\n0 10\n2 30\nEND\n")
            parser = ibl.SpectrumParser(start_marker="BEGIN", end_marker="END",
                        real_time=ibl.TimeField("Real", .001), live_time=ibl.TimeField("Live", .001))
            imported = ibl.read_instrument_spectrum(path, "A", parser)
            self.assertEqual((imported.real_time, imported.live_time), (2, 1.5))
            self.assertEqual(imported.spectrum.valid.tolist(), [True, False, True])
            result = ibl.FitObjective(self.problem()).best_result()
            ibl.write_fit_result(root/"result", result, provenance="unit-test")
            self.assertTrue((root/"result"/"sample.csv").is_file())
            with self.assertRaises(RuntimeError):
                ibl.write_fit_result(root/"result", result)

    def test_roi_policies_and_missing_data(self):
        p = self.problem()
        p.loss = "mse"
        p.selections = [ibl.SpectrumSelection("A", 2, (ibl.Region(0, 1, 1), ibl.Region(1, 2, 3)))]
        o = ibl.FitObjective(p)
        self.assertAlmostEqual(o.evaluate_batch([[120]])[0], 800)
        p.region_normalization = "weight_sum"
        self.assertAlmostEqual(ibl.FitObjective(p).evaluate_batch([[120]])[0], 200)
        p.selections = [ibl.SpectrumSelection("A", 1, (ibl.Region(0, 2), ibl.Region(1, 2)))]
        with self.assertRaises(ibl.FitError):
            ibl.FitObjective(p)
        p.allow_overlapping_regions = True
        p.region_normalization = "sum"
        self.assertAlmostEqual(ibl.FitObjective(p).evaluate_batch([[120]])[0], 300)
        p.selections = [ibl.SpectrumSelection("A", regions=(ibl.Region(0, 3),))]
        p.missing_observations_as_zero = True
        p.missing_prediction = "zero"
        self.assertAlmostEqual(ibl.FitObjective(p).evaluate_batch([[120]])[0], 200)

    def test_layer_properties_and_profile_links(self):
        p = self.problem()
        p.parameters = [ibl.LayerProperty(0, "porosity_fraction", (0, .5))]
        self.assertEqual(ibl.FitObjective(p).decode([.2]).sample.layers[0].porosity_fraction, .2)
        p.parameters = []
        p.profile = ibl.ProfileModel((ibl.ProfileLayer(p.initial_sample.layers[0],
                    (ibl.polynomial(1, -1), ibl.polynomial(0, 1)), 2),))
        p.profile_parameters = [ibl.ProfileParameter(0, 0, 1, (-1, -.1), "slope", ((0, 1, 1, -1),))]
        o = ibl.FitObjective(p)
        self.assertTrue(np.isfinite(o.evaluate_batch([[-.5]])).all())
        p.profile_parameters = [ibl.ProfileParameter(0, 0, 1, (-1, -.1), "slope", ((0, 1, 1, 1),))]
        with self.assertRaises(ibl.FitError):
            ibl.FitObjective(p)

    def test_cancel_and_worker_errors(self):
        result = ibl.fit(self.problem(), progress=lambda stats: True)
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.evaluations, 1)
        executor = ibl.AnalysisExecutor(1, lambda worker: Inventory())
        def stop(task, worker, stats):
            executor.request_stop()
            return True
        out = executor.run([ibl.AnalysisTask("a", self.problem()), ibl.AnalysisTask("b", self.problem())], progress=stop)
        self.assertEqual(out[0].result.status, "cancelled")
        self.assertIsNone(out[1].result)
        self.assertEqual(out[1].status, "cancelled")
        executor = ibl.AnalysisExecutor(1, lambda worker: Inventory())
        out = executor.run([ibl.AnalysisTask("a", self.problem(), {"max_evaluations": 1})],
                           progress=lambda *args: (_ for _ in ()).throw(ValueError("callback failed")))
        self.assertIn("callback failed", out[0].error)

    @unittest.skipUnless(engine().arrde_available(), "native optimizer is disabled")
    def test_native_pipeline(self):
        p = self.problem()
        p.data = ibl.MeasuredData(p.data.setup, [ibl.MeasuredSpectrum("A", [75, 75])])
        result = ibl.fit(p, initial_guesses=[[150]], stages=[dict(optimizer="de", max_evaluations=12, population_size=4, initial_guesses=[[150]]),
                                                dict(max_evaluations=12, population_size=4)])
        self.assertEqual(result.cost, 0)
        self.assertLessEqual(result.evaluations, 24)


if __name__ == "__main__":
    unittest.main()
