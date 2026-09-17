import pathlib
import sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
from ibeamlab import _ibeamlab_cpp as cpp

scaler = cpp.transforms.StandardScaler([1.0, 2.0], [2.0, 4.0])
assert np.allclose(scaler.apply(np.array([[3.0, 6.0]], np.float32)), [[1.0, 1.0]])
spectrum = cpp.simulator.Spectrum()
spectrum.label = "RBS"
spectrum.counts = np.array([1.0, 2.0], np.float32)
assert isinstance(spectrum.counts, np.ndarray)

assert cpp.model.ModelType.INVERSE != cpp.model.ModelType.FORWARD
assert hasattr(cpp.inference, "InverseModel")
assert hasattr(cpp.inference, "ForwardModel")
