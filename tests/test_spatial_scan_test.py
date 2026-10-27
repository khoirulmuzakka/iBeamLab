"""Invariants of the held-out Na2FePO4F synthetic experiment."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import generate_spatial_scan_test as test_scan


class HeldOutMaterialTests(unittest.TestCase):
    def test_all_three_held_out_profiles(self):
        t = test_scan.scan.sublayer_thicknesses()
        edges = np.r_[0, np.cumsum(t)]
        z = (edges[:-1]+edges[1:])/2
        r = 5*np.sqrt((np.arange(50)+.5)/50)
        for name, chem in test_scan.TEST_CHEMISTRIES.items():
            for cell in test_scan.cell_layout(chem):
                f = test_scan.array_composition(cell["center_x_mm"]+r[:, None], 0, z[None, :], chem)
                np.testing.assert_allclose(f.sum(axis=-1), 1, atol=1e-13)
                ion = f[..., test_scan.ELEMENTS.index(chem.mobile_ion)]
                x = chem.host_atoms*ion/(1-ion)
                self.assertGreaterEqual(x.min(), chem.x_min-1e-12, name)
                self.assertLessEqual(x.max(), chem.x_max+1e-12, name)
                mean = np.average(ion, axis=1, weights=t).mean()
                recovered = float(test_scan.scan.soc_from_mobile_fraction(chem, mean))
                self.assertAlmostEqual(recovered, cell["soc"], places=12, msg=name)

    def test_one_sodium_window_and_grid(self):
        chem = test_scan.CHEMISTRY
        np.testing.assert_allclose(test_scan.scan.stoichiometry_from_soc(chem, [0, 1]), [2, 1])
        self.assertEqual(test_scan.scan_grid()[0].shape, (20, 100))
        self.assertEqual(len(test_scan.CELLS), 5)
        self.assertEqual(chem.host_atoms, 7)

    def test_fractions_hosts_bounds_and_integrated_soc(self):
        t = test_scan.scan.sublayer_thicknesses()
        edges = np.r_[0, np.cumsum(t)]
        midpoints = (edges[:-1]+edges[1:])/2
        r = 5*np.sqrt((np.arange(60)+.5)/60)
        chem = test_scan.CHEMISTRY
        for cell in test_scan.CELLS:
            f = test_scan.array_composition(cell["center_x_mm"]+r[:, None], 0, midpoints[None, :])
            np.testing.assert_allclose(f.sum(axis=-1), 1, atol=1e-14)
            ion = f[..., test_scan.ELEMENTS.index("Na")]
            x = 7*ion/(1-ion)
            self.assertGreaterEqual(x.min(), 1-1e-13)
            self.assertLessEqual(x.max(), 2+1e-13)
            np.testing.assert_array_equal(f[..., test_scan.ELEMENTS.index("Al")], 0)
            for e, n in chem.host.items():
                np.testing.assert_allclose(f[..., test_scan.ELEMENTS.index(e)]/(1-ion), n/7, atol=1e-14)
            average = np.average(ion, axis=1, weights=t).mean()
            self.assertAlmostEqual(float(test_scan.scan.soc_from_mobile_fraction(chem, average)), cell["soc"], places=13)
        holder = test_scan.array_composition(28, 6, midpoints)
        np.testing.assert_array_equal(holder[:, test_scan.ELEMENTS.index("Al")], 1)

    def test_material_configuration_includes_fluorine_and_only_sodium_shape(self):
        base = test_scan.scan
        entries = []
        for e in base.ARRAY_ELEMENTS:
            for c, value in zip("Aabc", [.1, 0., 0., 1.]):
                entries.append([f"Erf_1_ {e}_{c}", value, 0., 1., "FREE" if c == "A" or e in ("Li", "Na") else "FIXED"])
        template = {"BatchResults": [{"Sample": "old"}], "Config": {
            "Analysis": {"Parameters": entries, "Parameter_constraints": []},
            "Layers": {"NumberOfLayers": 1, "Parameterization": ["Erf-polynomial"],
                       "Parameterization_Degree": [0],
                       "ElementsPerLayer": [", ".join(f"{e}:0.1" for e in base.ARRAY_ELEMENTS)]}}}
        config = test_scan.fit_configuration(template, ["new"], Path("new"))
        self.assertNotIn("BatchResults", config)
        params = config["Config"]["Analysis"]["Parameters"]
        self.assertEqual(sum(p[-1] == "FREE" for p in params), 9)
        self.assertTrue(any(p[0] == "Erf_1_ F_A" and p[-1] == "FREE" for p in params))
        for p in params:
            if p[0].endswith(("_a", "_b", "_c")):
                self.assertEqual(p[-1], "FREE" if " Na_" in p[0] else "FIXED")
        self.assertEqual(template["Config"]["Layers"]["ElementsPerLayer"],
                         [", ".join(f"{e}:0.1" for e in base.ARRAY_ELEMENTS)])


if __name__ == "__main__":
    unittest.main()
