"""Licensed local SIMNRA diagnostics and project-export integration check."""
from pathlib import Path
from dataclasses import replace
import sys
import tempfile
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if len(sys.argv)>1 and Path(sys.argv[1]).is_dir():
    sys.path.insert(0, str(Path(sys.argv.pop(1)).resolve()))
import ibeamlab as ibl
from ibeamlab import native
from ibeamlab.sample import experiment_from_core
from cpp.tests.simnra_xnra import load_reference


class SimnraDiagnosticsTests(unittest.TestCase):
    def test_elemental_spectra_and_export(self):
        reference = ROOT/"xnra/Ref_RBS_LiCOFNaAlSiPSTiMnFeCoNiCuH_nopu.xnra"
        raw, _, _ = load_reference(reference, native)
        x = experiment_from_core(raw)
        x = replace(x, setup=ibl.ExperimentalSetup([replace(x.setup.measurements[0], label="RBS")]))
        method = ibl.SimnraMethod("RBS", reference, x.setup.measurements[0].beam.particle)
        with ibl.SimnraSimulator([method], workers=1, fast_calculation=True, capture_elemental=True) as simulator:
            self.assertTrue(simulator.capabilities.elemental_spectra)
            self.assertTrue(simulator.capabilities.project_export)
            result = simulator.simulate(x)
            self.assertIsNone(result.failure)
            self.assertTrue(result.elemental_spectra["RBS"])
            self.assertTrue(all(np.isfinite(v).all() for v in result.elemental_spectra["RBS"].values()))
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory)/"fitted.xnra"
                simulator.export_project(x, target, measured=result["RBS"].counts)
                self.assertTrue(target.is_file())
                with ibl.SimnraSimulator([ibl.SimnraMethod("RBS", target, method.particle)],
                                        workers=1, fast_calculation=True) as restored:
                    reloaded = restored.simulate(x)
                    self.assertIsNone(reloaded.failure)
                    np.testing.assert_allclose(reloaded["RBS"].counts, result["RBS"].counts,
                                               rtol=1e-9, atol=1e-8)
                with self.assertRaises(RuntimeError):
                    simulator.export_project(x, target)


if __name__ == "__main__":
    unittest.main()
