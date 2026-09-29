"""Generate an NMC RBS/NRA dataset through the public Python API."""
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ibeamlab as ibl

SIMNRA_WORKERS = 8
METHODS = (
    ibl.SimnraMethod("RBS", ROOT / "xnra/Ref_RBS_LiCOFNaAlSiPSTiMnFeCoNiCuH_nopu.xnra"),
    ibl.SimnraMethod("NRA", ROOT / "xnra/Ref_NRA_LiCOFNaAlSiPSTiMnFeCoNiCuH_nopu.xnra"),
)
ELEMENTS = ("Li", "Ni", "Mn", "Co", "O")

def make_experiment(layers: int = 1) -> ibl.Experiment:
    sample = ibl.Sample([
        ibl.Layer(500_000 / layers, {element: 0.2 for element in ELEMENTS})
        for _ in range(layers)
    ])
    detectors = [
        ibl.Detector("RBS", ibl.Beam("H", 2974), ibl.LinearCalibration(2.63714),
                     resolution=20, particles_sr=1e12, real_time=0.001, live_time=0.001),
        ibl.Detector("NRA", ibl.Beam("H", 2974), ibl.LinearCalibration(7.55),
                     resolution=20, particles_sr=1e13, real_time=0.001, live_time=0.001),
    ]
    return ibl.Experiment(sample, detectors)

def make_study(layers: int = 1) -> ibl.GenerationStudy:
    parameters = [
        ibl.vary.concentration(element, layer=layer, bounds=(0, 1))
        for layer in range(layers) for element in ELEMENTS
    ]
    return ibl.GenerationStudy(make_experiment(layers), parameters, methods=METHODS)

def main() -> None:
    output = ROOT / "examples/datasets" / f"nmc_{datetime.now():%Y%m%d_%H%M%S_%f}"
    with ibl.SimnraSimulator(METHODS, workers=SIMNRA_WORKERS) as simulator:
        summary = make_study().generate(output, simulator=simulator, samples=1000,
                                        seed=1, batch_size=SIMNRA_WORKERS)
    print(summary)

if __name__ == "__main__":
    main()
