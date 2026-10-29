"""End-to-end checks against the new extension, never the legacy package."""
import sys
from pathlib import Path
import tempfile
import numpy as np
sys.path.insert(0, str(Path(sys.argv[1])))
sys.path.insert(0, str(Path(__file__).parents[2]))
from ibeamlab import native as ibl

def experiment():
    x = ibl.Experiment()
    species = ibl.Species(); species.element = 'Si'; species.concentration = 1
    layer = ibl.Layer(); layer.thickness = 100; layer.species = [species]
    sample = ibl.Sample(); sample.layers = [layer]; x.sample = sample
    beam = ibl.Beam(); beam.particle = 'H'; beam.energy = 2000
    measurement = ibl.Measurement(); measurement.label = 'RBS'; measurement.beam = beam
    setup = ibl.ExperimentalSetup(); setup.measurements = [measurement]; x.setup = setup
    x.validate()
    return x

x = experiment()
assert ibl.read_parameter(x, ibl.LayerThickness(0)) == 100
sim = ibl.DummySimulator(4)
assert len(sim.simulate(x).spectra[0].counts) == 4
counts = np.array([2., 4.]); edges = np.array([0., 1., 2.]); output = np.empty(2)
ibl.rebin_into(counts, edges, edges, output)
np.testing.assert_array_equal(output, counts)
try:
    ibl.rebin_into(counts.astype(np.float32), edges, edges, output)
except TypeError:
    pass
else:
    raise AssertionError('unexpected silent float64 conversion')
workspace = ibl.PileupWorkspace()
piled = np.empty(3); restored = np.empty(2)
ibl.apply_pileup_into(counts, piled, 10, 8, .01, workspace)
ibl.remove_pileup_into(piled, restored, 10, 8, .01)
np.testing.assert_allclose(restored, counts)
with tempfile.TemporaryDirectory(prefix='ibeamlab-python-test-') as directory:
    root = Path(directory)
    plan = ibl.GenerationPlan(); plan.reference = x
    plan.parameters = [ibl.Parameter('thickness', ibl.LayerThickness(0), '1e15 atoms/cm2')]
    progress = []
    summary = ibl.generate(plan=plan, rows=np.array([[100.], [200.], [300.]]), simulator=sim,
                           path=root/'dataset', progress=lambda done, total: progress.append((done, total)),
                           provenance='seed=123', records_per_shard=1)
    assert summary.accepted == 3 and progress[-1] == (3, 3)
    reader = ibl.DatasetReader(root/'dataset')
    records = list(reader)
    assert [r.parameters[0] for r in records] == [100, 200, 300]
    assert 'seed=123' in reader.metadata.provenance
    if len(sys.argv) > 2 and sys.argv[2] == 'ON':
        fixtures = Path(__file__).parent / 'fixtures'
        forward = ibl.ForwardModelMetadata(); forward.reference = x
        forward.input_parameters = plan.parameters
        forward.output_spectra = ibl.SpectraSchema([ibl.SpectrumSpec('RBS', 4)])
        metadata = ibl.ModelMetadata(forward)
        assert metadata.direction == ibl.Direction.FORWARD
        assert isinstance(metadata.model, ibl.ForwardModelMetadata)
        package = ibl.Package.from_onnx(fixtures/'forward.onnx', metadata)
        package.write(root/'forward.ibm')
        result = ibl.ForwardModel(ibl.Package.open(root/'forward.ibm')).predict(x)
        np.testing.assert_allclose(result.spectra[0].counts, [100, 200, 300, 400])
        inverse_spec = ibl.InverseModelMetadata(); inverse_spec.reference_setup = x.setup
        inverse_spec.input_spectra = forward.output_spectra
        inverse_spec.elements = ['Si', 'O']; inverse_spec.layers = 1; inverse_spec.uncertainty = True
        inverse = ibl.ModelMetadata(inverse_spec)
        assert inverse.direction == ibl.Direction.INVERSE
        assert isinstance(inverse.model, ibl.InverseModelMetadata)
        switching = ibl.ModelMetadata(forward)
        switching.model = inverse_spec
        assert switching.direction == ibl.Direction.INVERSE
        switching.validate()
        try:
            switching.direction = ibl.Direction.FORWARD
        except AttributeError:
            pass
        else:
            raise AssertionError('Direction must be derived from the model specification')
        model = ibl.InverseModel(ibl.Package.from_onnx(fixtures/'inverse.onnx', inverse))
        measured = ibl.InverseInput(); measured.setup = x.setup
        measured.spectra = [ibl.Spectrum('RBS', [1., 2., 3., 4.])]
        np.testing.assert_allclose(model.predict(measured).values, [10, 20])
        import torch
        class Forward(torch.nn.Module):
            def forward(self, values):
                return values * torch.tensor([[1., 2., 3., 4.]])
        original = Forward()
        exported = ibl.export_package(original, torch.ones(2, 1), metadata, root/'torch.ibm')
        assert original.training
        np.testing.assert_allclose(ibl.ForwardModel(exported).predict(x).spectra[0].counts,
                                   [100, 200, 300, 400])
        class Inverse(torch.nn.Module):
            def forward(self, values):
                total = values.sum(dim=1, keepdim=True)
                ones = torch.ones_like(total)
                return torch.cat((total, total*2, ones*.8, ones*.5, ones*.2, ones*.3), dim=1)
        inverse_export = ibl.export_package(Inverse(), torch.ones(2, 4), inverse, root/'inverse-torch.ibm')
        loaded_inverse = ibl.Package.open(root/'inverse-torch.ibm')
        assert isinstance(loaded_inverse.metadata.model, ibl.InverseModelMetadata)
        assert loaded_inverse.metadata.model.reference_setup.measurements[0].label == 'RBS'
        np.testing.assert_allclose(ibl.InverseModel(inverse_export).predict(measured).values, [10, 20])
        def multi_experiment(rbs_dose, nra_dose, reverse=False):
            value = experiment()
            rbs = value.setup.measurements[0]
            acquisition = ibl.Acquisition(); acquisition.particlesSr = rbs_dose
            rbs.acquisition = acquisition
            nra = ibl.Measurement(); nra.label = 'NRA'
            beam = ibl.Beam(); beam.particle = 'H'; beam.energy = 3000
            nra.beam = beam
            acquisition = ibl.Acquisition(); acquisition.particlesSr = nra_dose
            nra.acquisition = acquisition
            setup = ibl.ExperimentalSetup()
            setup.measurements = [nra, rbs] if reverse else [rbs, nra]
            value.setup = setup
            return value
        multi_ref = multi_experiment(1000, 2000)
        multi_request = multi_experiment(2000, 1000, reverse=True)
        multi_forward = ibl.ForwardModelMetadata(); multi_forward.reference = multi_ref
        multi_forward.input_parameters = plan.parameters
        multi_forward.output_spectra = ibl.SpectraSchema([ibl.SpectrumSpec('RBS', 4), ibl.SpectrumSpec('NRA', 3)])
        assert multi_forward.output_spectra.width() == 7 and multi_forward.output_spectra.offset('NRA') == 4
        multi_forward.postprocessing = [ibl.SpectrumProcessing('RBS', [ibl.ExposureScale(), ibl.Rebin()]),
                                        ibl.SpectrumProcessing('NRA', [ibl.MultiplyCounts(.5), ibl.ExposureScale(), ibl.Rebin()])]
        multi_package = ibl.Package.from_onnx(fixtures/'forward_multi.onnx', ibl.ModelMetadata(multi_forward))
        multi_package.write(root/'multi-forward.ibm')
        predicted = ibl.ForwardModel(ibl.Package.open(root/'multi-forward.ibm')).predict(multi_request)
        assert [s.measurement for s in predicted.spectra] == ['RBS', 'NRA']
        np.testing.assert_allclose(predicted.spectra[0].counts, [200, 400, 600, 800])
        np.testing.assert_allclose(predicted.spectra[1].counts, [125, 150, 175])
        multi_inverse = ibl.InverseModelMetadata(); multi_inverse.reference_setup = multi_ref.setup
        multi_inverse.input_spectra = multi_forward.output_spectra
        multi_inverse.elements = ['Si', 'O']; multi_inverse.layers = 1; multi_inverse.uncertainty = True
        multi_inverse.input_preparation = [ibl.SpectrumProcessing(label, [ibl.ExposureScale(), ibl.Rebin()]) for label in ['RBS', 'NRA']]
        multi_package = ibl.Package.from_onnx(fixtures/'inverse_multi.onnx', ibl.ModelMetadata(multi_inverse))
        multi_package.write(root/'multi-inverse.ibm')
        measured = ibl.InverseInput(); measured.setup = multi_request.setup
        measured.spectra = [ibl.Spectrum('NRA', [.5, 1, 1.5]), ibl.Spectrum('RBS', [2, 4, 6, 8])]
        np.testing.assert_allclose(ibl.InverseModel(ibl.Package.open(root/'multi-inverse.ibm')).predict(measured).values, [70, 140])
        scaling = ibl.ForwardModelMetadata(); scaling.reference = x; scaling.input_parameters = plan.parameters
        scaling.output_spectra = forward.output_spectra
        scaling.preprocessing = [ibl.MinMaxScale([0], [200])]
        scaling.postprocessing = [ibl.SpectrumProcessing('RBS', [ibl.MultiplyCounts(1000)])]
        scale_package = ibl.Package.from_onnx(fixtures/'forward.onnx', ibl.ModelMetadata(scaling))
        np.testing.assert_allclose(ibl.ForwardModel(scale_package).predict(x).spectra[0].counts, [500, 1000, 1500, 2000])
print('Independent Python API checks passed')
