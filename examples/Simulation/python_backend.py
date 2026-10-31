"""Synthetic Python simulator illustrating the public backend interface.

Run from the repository root after building/installing iBeamLab:
    python examples/Simulation/python_backend.py --output datasets/python_backend
"""
import argparse
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import ibeamlab as ibl


class PythonSimulator(ibl.Simulator):
    def __init__(self, channels=128):
        self.channels = np.arange(channels, dtype=np.float64)

    def simulate(self, experiment):
        # Replace this calculation with your simulator or external program call.
        amount = sum(layer.thickness for layer in experiment.sample.layers)
        return {measurement.label: amount * np.exp(-0.5 * (
            (self.channels - measurement.detector.geometry.scattering_angle / 2) / 8) ** 2)
            for measurement in experiment.setup.measurements}

    @property
    def provenance(self):
        return "python-backend-example; synthetic-gaussian; version=1"

    # A vectorized backend may override simulate_batch(experiments) and return
    # one mapping or SimulationResult per input. The inherited default calls
    # simulate() and turns per-experiment exceptions into failure records.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/python_backend"))
    parser.add_argument("--extension-dir", type=Path)
    args = parser.parse_args()
    if args.extension_dir:
        sys.path.insert(0, str(args.extension_dir.resolve()))
    measurement = ibl.Measurement("RBS", ibl.Beam("4He", 2000), ibl.Detector(
        geometry=ibl.Geometry(incidence_angle=0, exit_angle=15, scattering_angle=165)))
    experiment = ibl.Experiment(ibl.Sample([ibl.Layer(100, {"Si": 1})]),
                                ibl.ExperimentalSetup([measurement]))
    plan = ibl.GenerationPlan(experiment, [ibl.vary.layer_thickness(layer=0, bounds=(50, 200))])
    backend = PythonSimulator()
    summary = plan.generate(plan.sample_uniform(8, seed=42), backend, args.output,
                            batch_size=4, progress=lambda done, total: print(f"{done}/{total}"))
    print(summary)


if __name__ == "__main__": main()
