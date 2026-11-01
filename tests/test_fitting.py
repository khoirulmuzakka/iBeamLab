"""Small physical fitting checks; no training or licensed simulator is required."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if len(sys.argv)>1 and Path(sys.argv[1]).is_dir():
    sys.path.insert(0, str(Path(sys.argv.pop(1)).resolve()))
ONNX = len(sys.argv)>1 and sys.argv[1] == "ON"
if len(sys.argv)>1 and sys.argv[1] in ("ON", "OFF"):
    sys.argv.pop(1)
import ibeamlab as ibl
from ibeamlab._engine import engine


def experiment():
    return ibl.Experiment(ibl.Sample([ibl.Layer(100, {"Si": .5, "O": .5})]),
        ibl.ExperimentalSetup([ibl.Measurement("RBS", ibl.Beam("H", 2000))]))


class BatchBackend(ibl.Simulator):
    def __init__(self):
        self.batches = []
    def simulate(self, x):
        raise AssertionError("population objective must use batch prediction")
    def simulate_batch(self, experiments):
        self.batches.append(len(experiments))
        return [{m.label: x.sample.layers[0].thickness * np.arange(1,5,dtype=float)
                 for m in x.setup.measurements} for x in experiments]


def problem(predictor=None, counts=None, **kwargs):
    x=experiment()
    return ibl.FitProblem(ibl.MeasuredData(x.setup,[ibl.MeasuredSpectrum("RBS", counts if counts is not None else [150,300,450,600])]),
        x.sample,[ibl.vary.layer_thickness(layer=0,bounds=(10,200))],predictor or BatchBackend(),**kwargs)


class FittingTests(unittest.TestCase):
    def test_sparse_reader_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"data.txt"
            path.write_text("# detector\n1 4\n3 0\n")
            observed=ibl.read_spectrum(path,measurement="RBS",channel_base=1)
            np.testing.assert_array_equal(observed.counts,[4,0,0])
            np.testing.assert_array_equal(observed.valid,[True,False,True])
            self.assertFalse(observed.counts.flags.writeable)
            path.write_text('channel,counts,var\n"0","4","2"\n"2","0","3"\n')
            observed=ibl.read_spectrum(path,measurement="RBS",format="csv",header=True,variance_column=2)
            np.testing.assert_array_equal(observed.variances,[2,0,3])
            path.write_text("1\n2\n3\n")
            np.testing.assert_array_equal(ibl.read_spectrum(path,measurement="RBS",format="counts").counts,[1,2,3])
            path.write_text("0 1\n0 2\n")
            with self.assertRaisesRegex(ibl.DataReadError,r"data.txt:2:.*duplicate"):
                ibl.read_spectrum(path,measurement="RBS")

    def test_custom_reader(self):
        ibl.register_spectrum_reader("fitting_test",lambda path,measurement:ibl.MeasuredSpectrum(measurement,[3,4]))
        np.testing.assert_array_equal(ibl.read_spectrum("unused",measurement="RBS",format="fitting_test").counts,[3,4])
        with self.assertRaises(ValueError):
            ibl.register_spectrum_reader("fitting_test",lambda *args:None)

    def test_batch_chi_square_invalid_and_reduced_population(self):
        p=problem();objective=ibl.FitObjective(p)
        rows=np.array([[100.],[150.],[210.],[120.]])
        out=np.empty(4)
        self.assertIs(objective.evaluate_batch(rows,out=out),out)
        data=np.array([150,300,450,600])
        self.assertAlmostEqual(out[0],np.sum((data-np.array([100,200,300,400]))**2/(1+data)))
        self.assertEqual(out[1],0)
        self.assertTrue(np.isinf(out[2]))
        objective.evaluate_batch([[140.]])
        self.assertEqual(p.predictor.batches,[1,3,1])
        result=objective.best_result()
        self.assertEqual(result.parameters[next(iter(result.parameters))],150)
        self.assertEqual(result.evaluations,6)
        self.assertEqual(result.invalid,1)
        np.testing.assert_array_equal(result.prediction["RBS"].counts,data)

    def test_mask_variance_roi_and_measurement_weights(self):
        p=problem(counts=[110,200,330,400])
        p.data=ibl.MeasuredData(p.data.setup,[ibl.MeasuredSpectrum("RBS",[110,200,330,400],[1,0,1,1],[2,2,3,4])])
        p.selections=[ibl.SpectrumSelection("RBS",2,[ibl.Region(0,3,3)])]
        objective=ibl.FitObjective(p)
        self.assertAlmostEqual(objective.best_result().cost,6*(100/111+900/331))
        p.loss="variance_chi_square"
        self.assertAlmostEqual(ibl.FitObjective(p).best_result().cost,6*(100/2+900/3))
        p.selections=[ibl.SpectrumSelection("RBS",1,[ibl.Region(0,3),ibl.Region(2,4)])]
        with self.assertRaisesRegex(ibl.FitError,"overlapping"):
            ibl.FitObjective(p)

    def test_areal_density_and_composition(self):
        p=problem();p.parameters=[ibl.ArealDensity(0,"Si",(0,200)),ibl.ArealDensity(0,"O",(0,200))]
        o=ibl.FitObjective(p);x=o.decode([30,70]);self.assertEqual(x.sample.layers[0].thickness,100)
        self.assertAlmostEqual(x.sample.layers[0].species[0].concentration,.3)
        p.parameters=[ibl.vary.layer_thickness(layer=0,bounds=(10,200)),ibl.ArealDensity(0,"Si",(0,200))]
        with self.assertRaisesRegex(ibl.FitError,"conflicts"):
            ibl.FitObjective(p)
        p.parameters=[];p.compositions=[ibl.Composition(0,["Si","O"],[(.2,.7),(.3,.8)])]
        o=ibl.FitObjective(p);self.assertEqual(o.dimension,1)
        for coordinate,expected in [(0,.2),(1,.7),(.4,.4)]:
            x=o.decode([coordinate]);self.assertAlmostEqual(x.sample.layers[0].species[0].concentration,expected)
            self.assertAlmostEqual(sum(s.concentration for s in x.sample.layers[0].species),1)

    def test_ties_fixed_and_invalid_initial(self):
        p=problem();p.initial_sample=ibl.Sample([ibl.Layer(100,{"Si":1}),ibl.Layer(100,{"O":1})])
        p.parameters=[ibl.TiedParameter(ibl.vary.layer_thickness(layer=0,bounds=(10,200)),[ibl.LayerThickness(1)])]
        x=ibl.FitObjective(p).decode([80]);self.assertEqual([l.thickness for l in x.sample.layers],[80,80])
        p.parameters=[ibl.vary.layer_thickness(layer=0,bounds=(100,100))]
        o=ibl.FitObjective(p);self.assertEqual(o.dimension,0)
        self.assertEqual(ibl.fit(o).status,"fixed")
        p.parameters=[ibl.vary.layer_thickness(layer=0,bounds=(1,99))]
        with self.assertRaisesRegex(ibl.FitError,"outside bounds"):
            ibl.FitObjective(p)

    def test_external_optimizer_and_cancellation(self):
        p=problem()
        result=ibl.fit(p,optimizer=lambda objective,options:objective.evaluate_batch([[150],[80]]))
        self.assertEqual(result.cost,0)
        result=ibl.fit(problem(),progress=lambda p:True)
        self.assertEqual(result.status,"cancelled")
        self.assertEqual(result.evaluations,1)
        def interrupt(objective, options):
            objective.evaluate_batch([[150]])
            raise KeyboardInterrupt
        result=ibl.fit(problem(),optimizer=interrupt)
        self.assertEqual(result.status,"cancelled")
        self.assertEqual(result.cost,0)

    def test_detector_labels_and_unobserved_tail(self):
        p=problem();second=replace(p.data.setup.measurements[0],label="NRA")
        p.data=ibl.MeasuredData(ibl.ExperimentalSetup([p.data.setup.measurements[0],second]),[
            ibl.MeasuredSpectrum("NRA",[110,220]),
            ibl.MeasuredSpectrum("RBS",[150,300,450,600])])
        p.selections=[ibl.SpectrumSelection("NRA",2)]
        result=ibl.FitObjective(p).best_result()
        expected=2*(100/111+400/221)+sum((np.array([150,300,450,600])-np.array([100,200,300,400]))**2/(1+np.array([150,300,450,600])))
        self.assertAlmostEqual(result.cost,expected)
        self.assertEqual([c.measurement for c in result.contributions],["NRA","RBS"])
        self.assertEqual([c.channels for c in result.contributions],[2,4])
        p.selections=[]
        p.data=ibl.MeasuredData(problem().data.setup,[ibl.MeasuredSpectrum("RBS",[1,2,3,4,5])])
        with self.assertRaisesRegex(ibl.FitError,"cover selected"):
            ibl.FitObjective(p)

    def test_physical_invalid_calibration_and_zero_ad(self):
        p=problem();p.parameters=[ibl.vary.calibration_quadratic("RBS",bounds=(-1,1))]
        o=ibl.FitObjective(p)
        scores=o.evaluate_batch([[-1],[0]])
        self.assertTrue(np.isinf(scores[0]))
        self.assertTrue(np.isfinite(scores[1]))
        self.assertEqual(p.predictor.batches,[1,1])
        p.data=ibl.MeasuredData(p.data.setup,[ibl.MeasuredSpectrum("RBS",[1,2])])
        o=ibl.FitObjective(p)
        self.assertTrue(np.isinf(o.evaluate_batch([[-.2]])[0]))
        p=problem();p.parameters=[ibl.ArealDensity(0,"Si",(0,100)),ibl.ArealDensity(0,"O",(0,100))]
        x=ibl.FitObjective(p).decode([0,0])
        self.assertEqual(x.sample.layers[0].thickness,0)
        self.assertEqual([s.concentration for s in x.sample.layers[0].species],[.5,.5])

    def test_bound_aware_composition_population(self):
        p=problem();p.parameters=[]
        p.initial_sample=ibl.Sample([ibl.Layer(100,{"Si":.2,"O":.3,"C":.5})])
        p.compositions=[ibl.Composition(0,["Si","O","C"],[(.1,.4),(.2,.5),(.2,.6)])]
        o=ibl.FitObjective(p)
        for row in np.random.default_rng(42).uniform(size=(30,2)):
            values=[species.concentration for species in o.decode(row).sample.layers[0].species]
            self.assertAlmostEqual(sum(values),1)
            for value,bounds in zip(values,p.compositions[0].bounds):
                self.assertGreaterEqual(value+1e-12,bounds[0])
                self.assertLessEqual(value-1e-12,bounds[1])

    def test_cancel_after_population_and_separate_parallel_fits(self):
        if not engine().arrde_available():
            self.skipTest("ARRDE disabled")
        result=ibl.fit(problem(),max_evaluations=200,population_size=16,
                       progress=lambda p:p.evaluations>1)
        self.assertEqual(result.status,"cancelled")
        self.assertGreater(result.evaluations,1)
        self.assertEqual(result.history[-1].best_cost,result.cost)
        from concurrent.futures import ThreadPoolExecutor
        def solve(target):
            return ibl.fit(problem(counts=np.arange(1,5)*target),max_evaluations=300,population_size=12)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(solve,[120,160]))
        for result,target in zip(results,[120,160]):
            self.assertAlmostEqual(result.sample.layers[0].thickness,target,delta=.1)

    def test_progress_owns_snapshots_and_callback_interrupt(self):
        if not engine().arrde_available():
            self.skipTest("ARRDE disabled")
        snapshots=[]
        result=ibl.fit(problem(),max_evaluations=80,population_size=12,progress=snapshots.append)
        self.assertEqual(snapshots[0].evaluations,1)
        self.assertGreater(snapshots[-1].evaluations,1)
        self.assertLessEqual(snapshots[-1].best_cost,snapshots[0].best_cost)
        def interrupted(progress):
            if progress.evaluations>1:
                raise KeyboardInterrupt
        result=ibl.fit(problem(),max_evaluations=80,population_size=12,progress=interrupted)
        self.assertEqual(result.status,"cancelled")

    def test_arrde_fit(self):
        if not engine().arrde_available():
            self.skipTest("ARRDE disabled")
        p=problem()
        result=ibl.fit(p,max_evaluations=400,population_size=16,seed=42)
        self.assertLess(result.cost,1e-3)
        self.assertLessEqual(result.evaluations,400)
        self.assertTrue(any(n>1 for n in p.predictor.batches))
        self.assertTrue(result.history)

    @unittest.skipUnless(ONNX,"requires ONNX build")
    def test_onnx_population_and_metadata_permissions(self):
        x=experiment()
        metadata=ibl.ModelMetadata(ibl.ForwardModelMetadata(x.setup,
            ibl.ParameterInputSchema(x.sample,[ibl.ParameterSpec("thickness",ibl.LayerThickness(0))]),
            ibl.SpectraSchema([ibl.SpectrumSpec("RBS",4)])))
        model=ibl.ForwardModel(ibl.Package.from_onnx(ROOT/"cpp/tests/fixtures/forward.onnx",metadata))
        o=ibl.FitObjective(problem(model))
        np.testing.assert_allclose(o.evaluate_batch([[100],[150],[120]]),
            [np.sum((np.array([150,300,450,600])-t*np.arange(1,5))**2/(1+np.array([150,300,450,600]))) for t in [100,150,120]])
        np.testing.assert_array_equal(o.evaluate_batch([[150]]),[0])
        p=problem(model);p.parameters=[ibl.vary.beam_energy("RBS",bounds=(1900,2100))]
        with self.assertRaisesRegex(ibl.FitError,"cannot vary"):
            ibl.FitObjective(p)


if __name__ == "__main__":
    unittest.main()
