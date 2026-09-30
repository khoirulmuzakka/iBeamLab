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
