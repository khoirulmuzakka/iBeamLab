import pathlib
import sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
from ibeamlab import _ibeamlab_cpp as cpp

scaler = cpp.transforms.StandardScaler([1.0, 2.0], [2.0, 4.0])
assert np.allclose(scaler.apply(np.array([[3.0, 6.0]], np.float32)), [[1.0, 1.0]])
normalizer = cpp.transforms.LayerwiseConcentrationNormalizer(6, [[1, 2], [4, 5]])
parameter_scaler = cpp.transforms.ParameterBoundMinMaxScaler(
    [0.0] * 6, [1000.0, 1.0, 1.0, 100000.0, 1.0, 1.0]
)
pipeline = cpp.transforms.TransformPipeline()
pipeline.add(normalizer)
pipeline.add(parameter_scaler)
transformed = pipeline.apply(
    np.array([[100.0, 2.0, 6.0, 50000.0, -1.0, 3.0]], np.float32)
)
assert np.allclose(transformed, [[0.1, 0.25, 0.75, 0.5, 0.0, 1.0]])
spectrum = cpp.simulator.Spectrum()
spectrum.label = "RBS"
spectrum.counts = np.array([1.0, 2.0], np.float32)
assert isinstance(spectrum.counts, np.ndarray)

assert cpp.model.ModelType.INVERSE != cpp.model.ModelType.FORWARD
assert hasattr(cpp.inference, "InverseModel")
assert hasattr(cpp.inference, "ForwardModel")

from ibeamlab.spectrum import pileup, remove_pileup
original = [0.0, 10.0, 120.0, 45.0, 0.0, 3.0]
measured = pileup(original, 10.0, 8.0, 4e-4)
np.testing.assert_allclose(remove_pileup(measured, 10.0, 8.0, 4e-4), original, atol=1e-10)

# Exercise version 3 EDP packaging, native bindings, and the public Python result.
import tempfile
from pathlib import Path
from ibeamlab import InverseModel, Spectrum

metadata = cpp.model.ModelMetadata()
metadata.model_type = cpp.model.ModelType.INVERSE
metadata.input_dimension = 2
metadata.output_dimension = 1
input_transform = cpp.model.TransformSpec()
input_transform.input_dimension = 2
metadata.input_transform = input_transform
output_transform = cpp.model.TransformSpec()
output_transform.input_dimension = 1
metadata.output_transform = output_transform
sample_template = cpp.sample.SampleModel()
layer = cpp.sample.Layer()
layer.thickness = 1
species = cpp.sample.Species()
species.element = "Si"
species.concentration = 1
layer.species = [species]
sample_template.layers = [layer]
setup_template = cpp.sample.ExperimentalSetup()
detector = cpp.sample.Detector()
detector.label = "RBS"
beam = cpp.sample.Beam()
beam.particle = "He"
beam.energy = 2
detector.beam = beam
setup_template.detectors = [detector]
inverse_metadata = cpp.model.InverseModelMetadata()
inverse_metadata.need_pileup_subtraction = True
inverse_metadata.sample_template = sample_template
inverse_metadata.setup_template = setup_template
spectrum_spec = cpp.model.SpectrumSpec()
spectrum_spec.label = "RBS"
spectrum_spec.length = 2
inverse_metadata.input_spectra = [spectrum_spec]
edp_spec = cpp.model.EdpSpec()
edp_spec.max_layers = 1
edp_spec.elements = ["Si"]
inverse_metadata.output_edp = edp_spec
metadata.inverse = inverse_metadata
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "inverse.zip"
    package = cpp.model.ModelPackage.from_onnx(
        Path(__file__).parent / "data" / "model-package" / "model.onnx", metadata)
    package.write(path)
    assert cpp.model.ModelPackage.open(path).metadata.format_version == 3
    assert cpp.model.ModelPackage.open(path).metadata.inverse.need_pileup_subtraction
    model = InverseModel(path)
    prediction = model.predict_prepared({"RBS": Spectrum("RBS", [1, 2])})
    assert prediction.elements == ("Si",)
    assert prediction.unit == "1e15 atoms/cm2"
    np.testing.assert_allclose(prediction.values, [[9]])
    assert prediction.edp.to_sample(sample_template).layers[0].thickness == 9
