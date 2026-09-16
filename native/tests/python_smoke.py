import gc
import pathlib
import sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
from ibeamlab import _ibeamlab_cpp as cpp

scaler = cpp.preprocessing.StandardScaler([1.0, 2.0], [2.0, 4.0])
assert np.allclose(scaler.apply(np.array([[3.0, 6.0]], np.float32)), [[1.0, 1.0]])
spectrum = cpp.simulator.Spectrum()
spectrum.label = "RBS"
spectrum.counts = np.array([1.0, 2.0], np.float32)
assert isinstance(spectrum.counts, np.ndarray)

package = cpp.model.ModelPackage.open(pathlib.Path(__file__).parent / "data/model-package.zip")
for _ in range(10):
    engine = cpp.inference.OnnxInferenceEngine(package)
    result = engine.predict([[spectrum]])
    assert abs(result[0].values[0].value - 9.0) < 1e-5
    del engine
    gc.collect()
