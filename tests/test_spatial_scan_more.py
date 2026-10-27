"""Physical invariants of the eight-material synthetic scan (no SIMNRA)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import generate_spatial_scan_more as scan


class SpatialBatteryTests(unittest.TestCase):
    def test_layout_and_holder(self):
        self.assertEqual(len(scan._ARRAY_CELLS), 40)
        self.assertEqual((scan.HOLDER_WIDTH_MM, scan.HOLDER_HEIGHT_MM), (56, 89))
        x, y = scan.array_scan_grid(100)
        self.assertEqual(x.shape, (160, 100))
        self.assertEqual(y.shape, (160, 100))
        self.assertEqual(scan.array_scan_grid(10)[0].shape, (16, 10))
        f = scan.array_composition(x, y, 0)
        np.testing.assert_allclose(f.sum(axis=-1), 1, atol=1e-14)
        self.assertTrue(np.all(f >= 0))
        holder = scan._array_cell_ids(x, y) == -1
        np.testing.assert_array_equal(f[..., scan.ARRAY_ELEMENTS.index("Al")][holder], 1)

    def test_windows_roundtrip_bounds_and_ion_separation(self):
        for chem in scan.CHEMISTRIES:
            self.assertGreater(chem.x_max, chem.x_min)
            self.assertGreaterEqual(chem.x_min, 0)
            soc = np.linspace(0, 1, 51)
            f = scan.mobile_fraction_from_stoichiometry(chem, scan.stoichiometry_from_soc(chem, soc))
            np.testing.assert_allclose(scan.soc_from_mobile_fraction(chem, f), soc, atol=1e-14)
        for cell in scan._ARRAY_CELLS:
            chem = scan.CHEMISTRY_BY_NAME[cell["chemistry"]]
            r = np.linspace(0, scan.CELL_DIAMETER_MM/2, 41)
            z = np.linspace(0, scan.MAX_DEPTH, 51)
            f = scan.array_composition(cell["center_x_mm"]+r[:, None], cell["center_y_mm"], z[None, :])
            np.testing.assert_allclose(f.sum(axis=-1), 1, atol=1e-14)
            ion = f[..., scan.ARRAY_ELEMENTS.index(chem.mobile_ion)]
            x = chem.host_atoms * ion/(1-ion)
            self.assertGreaterEqual(x.min(), chem.x_min-1e-13)
            self.assertLessEqual(x.max(), chem.x_max+1e-13)
            other = "Na" if chem.mobile_ion == "Li" else "Li"
            np.testing.assert_array_equal(f[..., scan.ARRAY_ELEMENTS.index(other)], 0)
            np.testing.assert_array_equal(f[..., scan.ARRAY_ELEMENTS.index("Al")], 0)
            for element, atoms in chem.host.items():
                np.testing.assert_allclose(f[..., scan.ARRAY_ELEMENTS.index(element)]/(1-ion),
                                           atoms/chem.host_atoms, atol=1e-14)

    def test_integrated_soc_all_disks(self):
        thicknesses = scan.sublayer_thicknesses()
        edges = np.r_[0, np.cumsum(thicknesses)]
        z = (edges[:-1]+edges[1:])/2
        r = scan.CELL_DIAMETER_MM/2 * np.sqrt((np.arange(50)+0.5)/50)
        self.assertAlmostEqual(thicknesses.sum(), scan.MAX_DEPTH)
        for cell in scan._ARRAY_CELLS:
            chem = scan.CHEMISTRY_BY_NAME[cell["chemistry"]]
            f = scan.array_composition(cell["center_x_mm"]+r[:, None], cell["center_y_mm"],
                                       z[None, :])[..., scan.ARRAY_ELEMENTS.index(chem.mobile_ion)]
            mean = np.average(f, axis=1, weights=thicknesses).mean()
            self.assertAlmostEqual(float(scan.soc_from_mobile_fraction(chem, mean)), cell["soc"], places=13)

    def test_blend_and_capacity_conversion(self):
        c = scan.CHEMISTRY_BY_NAME["Silicon-carbon"]
        mass = scan.ATOMIC_MASS["Si"] + c.host["C"]*scan.ATOMIC_MASS["C"]
        self.assertAlmostEqual(scan.ATOMIC_MASS["Si"]/mass, 0.10)
        self.assertAlmostEqual(c.x_max, 3.75+c.host["C"]/6)
        hard = scan.CHEMISTRY_BY_NAME["Hard carbon"]
        self.assertAlmostEqual(hard.x_max*scan.FARADAY_MAH_PER_MOL/scan.ATOMIC_MASS["C"], 300)

    def test_invalid_coordinates_and_soc(self):
        for xyz in [(np.nan, 0, 0), (0, 0, -1), (100, 0, 0)]:
            with self.assertRaises(ValueError):
                scan.array_composition(*xyz)
        for soc in [-0.1, 1.1, np.nan]:
            with self.assertRaises(ValueError):
                scan.stoichiometry_from_soc(scan.CHEMISTRIES[0], soc)


if __name__ == "__main__":
    unittest.main()
