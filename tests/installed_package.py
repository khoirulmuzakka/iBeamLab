"""Smoke test run with python -I against an installed wheel, outside source imports."""
from pathlib import Path
import gc
import sys
import tempfile
import numpy as np
import ibeamlab as ibl

root = Path(__file__).resolve().parents[1]
assert Path(ibl.__file__).resolve().parent != root / "ibeamlab", "source package masked the wheel"
assert "torch" not in sys.modules
x = ibl.Experiment(ibl.Sample([ibl.Layer(100, {"Si": 1})]),
    ibl.ExperimentalSetup([ibl.Measurement("RBS", ibl.Beam("H", 2000))]))
result = ibl.DummySimulator(4).simulate(x)
counts = result["RBS"].counts
expected = counts.copy()
del result
gc.collect()
np.testing.assert_array_equal(counts, expected)
assert not counts.flags.writeable
with tempfile.TemporaryDirectory() as tmp:
    plan = ibl.GenerationPlan(x, [ibl.vary.layer_thickness(layer=0, bounds=(50, 150))])
    rows = plan.sample_uniform(3, seed=42)
    path = Path(tmp) / "data"
    plan.generate(rows, ibl.DummySimulator(4), path)
    dataset = ibl.open_dataset(path)
    np.testing.assert_array_equal(dataset.parameters, rows)
    assert dataset.spectra("RBS").shape == (3, 4)
    metadata = ibl.ModelMetadata(ibl.ForwardModelMetadata(x, plan.parameters,
        ibl.SpectraSchema([ibl.SpectrumSpec("RBS", 4)])))
    package = ibl.Package.from_onnx(root / "cpp/tests/fixtures/forward.onnx", metadata)
    package.write(Path(tmp) / "forward.ibm")
    loaded = ibl.Package.open(Path(tmp) / "forward.ibm")
    assert loaded.metadata.model.reference == x
    if "--onnx" in sys.argv:
        predicted = ibl.ForwardModel(loaded).predict(x)
        np.testing.assert_allclose(predicted["RBS"].counts, [100, 200, 300, 400])
print(f"Installed package verified: {ibl.__file__}")
