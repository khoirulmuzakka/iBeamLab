"""Small profile/discretization and physical-fit regression cases."""
import sys
from pathlib import Path
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if len(sys.argv)>1 and Path(sys.argv[1]).is_dir():
    sys.path.insert(0, str(Path(sys.argv.pop(1)).resolve()))
import ibeamlab as ibl


class ProfileTests(unittest.TestCase):
    def model(self, spacing="linear"):
        return ibl.ProfileModel((ibl.ProfileLayer(ibl.Layer(100, {"Si": .5, "O": .5}),
                    (ibl.polynomial(1, -1), ibl.polynomial(0, 1)), 2, spacing),))

    def test_reference_functions_and_midpoints(self):
        self.assertEqual(ibl.constant(-.4).evaluate(.2), .4)
        self.assertEqual(ibl.polynomial(1, -1).evaluate(25, width=100), .75)
        self.assertAlmostEqual(ibl.erf_polynomial(.5, .2, .4, .1, .3).evaluate(.4), .56)
        sample = self.model().discretize()
        self.assertEqual(len(sample.layers), 2)
        self.assertEqual(sample.layers[0].thickness, 50)
        np.testing.assert_allclose([l.species[0].concentration for l in sample.layers], [.75, .25])

    def test_logarithmic_conservation_and_topology(self):
        sample = self.model("logarithmic").discretize()
        self.assertAlmostEqual(sum(l.thickness for l in sample.layers), 100)
        self.assertLess(sample.layers[0].thickness, sample.layers[1].thickness)
        grid = ibl.Discretizer(self.model())
        self.assertEqual(len(grid.apply(self.model()).layers), 2)
        with self.assertRaises(ValueError):
            grid.apply(self.model("logarithmic"))

    def test_invalid_profile(self):
        with self.assertRaises(ValueError):
            ibl.erf_polynomial(.5, .2, .4, 0).to_core()
        with self.assertRaises(ValueError):
            ibl.ProfileModel((ibl.ProfileLayer(ibl.Layer(100, {"Si": 1}), (ibl.constant(0),)),)).discretize()
        with self.assertRaises(ValueError):
            ibl.ProfileModel((ibl.ProfileLayer(ibl.Layer(100, {"Si": 1}), (), 2),)).to_core()


if __name__ == "__main__":
    unittest.main()
