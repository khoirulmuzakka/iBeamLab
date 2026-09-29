"""Execute one simulation exclusively through the public Python API."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))
import ibeamlab as ibl

def main(reference_file: Path) -> None:
    experiment = ibl.Experiment(
        ibl.Sample([
            ibl.Layer(123.5, [
                ibl.Species("C", 0.35, (ibl.Isotope(12, 12.0, 0.9), ibl.Isotope(13, 13.0, 0.1))),
                ibl.Species("O", 0.65),
            ], roughness=4.25, porosity_fraction=0.12, pore_diameter=7.5),
            ibl.Layer(4567, {"Si": 0.8, "O": 0.2}),
        ]),
        [ibl.Detector("RBS", ibl.Beam("He", 2345, 8.5),
                      ibl.LinearCalibration(2.75, -3.5, -1.25e-5),
                      resolution=17.5, particles_sr=7.25e10,
                      real_time=91, live_time=87)],
    )
    method = ibl.SimnraMethod("RBS", reference_file.resolve())
    with ibl.SimnraSimulator([method]) as simulator:
        result = simulator.simulate(experiment)
    assert not result.failed
    assert result["RBS"].counts.ndim == 1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: simnra_python_integration.py REFERENCE.xnra")
    main(Path(sys.argv[1]))
