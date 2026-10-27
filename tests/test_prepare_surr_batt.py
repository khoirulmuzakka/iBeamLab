"""Material restrictions and preserved optimization controls in AutoNRA configs."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import prepare_surr_batt as prepare


def template():
    parameters = [["IBA_beamEnergy", 2974.0, 2655.0, 3245.0, "FIXED"],
                  ["Thickness_1", 100000.0, 90000.0, 110000.0, "FIXED"]]
    for element in prepare.scan.ARRAY_ELEMENTS:
        for coefficient in "Aabc":
            parameters.append([f"Erf_1_ {element}_{coefficient}", 0.1, 0.0, 1.0, "FREE"])
    return {"BatchResults": [{"Sample": "old-result"}], "Config": {
        "Analysis": {"Parameter_constraints": [], "Parameters": parameters,
                     "Sample_list": ["old-result"], "Number_of_samples": 1,
                     "Spectrum_dir": "old", "Result_dir": "old", "Restart_mode": "Resume"},
        "Layers": {"NumberOfLayers": 1, "Parameterization": ["Erf-polynomial"],
                   "Parameterization_Degree": [0], "N_sublayers": [7],
                   "ElementsPerLayer": [", ".join(f"{e}:0.1" for e in prepare.scan.ARRAY_ELEMENTS)]},
        "Fit": {"Pipeline": ["ARRDE"], "MaxEvals": [10000], "InitialValues": "Combined"},
        "IBA": {"Methods": {"IBA": {"Simulator": "Surrogate", "ROIs": [[100, 900, 1]]}}},
        "Optimizer": {"Methods": {"ARRDE": {"population_size": 0}}},
    }}


class MaterialConfigurationTests(unittest.TestCase):
    def test_allowed_elements_and_only_mobile_shape_freedom(self):
        for chem in prepare.scan.CHEMISTRIES:
            result = prepare.material_configuration(template(), chem, ["new-sample"], Path("new") / chem.name)
            self.assertNotIn("BatchResults", result)
            config = result["Config"]
            labels = {s.split(":")[0].strip() for s in config["Layers"]["ElementsPerLayer"][0].split(",")}
            self.assertEqual(labels, {chem.mobile_ion, "Al", *chem.host})
            for p in config["Analysis"]["Parameters"]:
                m = prepare.PARAMETER.fullmatch(p[0])
                if m:
                    element, coefficient = m.groups()
                    self.assertIn(element, labels)
                    expected = "FREE" if coefficient == "A" or element == chem.mobile_ion else "FIXED"
                    self.assertEqual(p[-1], expected)
            self.assertEqual(config["Analysis"]["Sample_list"], ["new-sample"])
            self.assertEqual(config["Analysis"]["Restart_mode"], "New")

    def test_optimizer_objective_and_template_are_preserved(self):
        original = template()
        before = copy.deepcopy(original)
        result = prepare.material_configuration(original, prepare.scan.CHEMISTRIES[0], ["x"], Path("new"))
        self.assertEqual(original, before)
        for key in ("Fit", "IBA", "Optimizer"):
            self.assertEqual(result["Config"][key], original["Config"][key])
        for parameter in result["Config"]["Analysis"]["Parameters"]:
            source = next(p for p in original["Config"]["Analysis"]["Parameters"] if p[0] == parameter[0])
            self.assertEqual(source[:4], parameter[:4])
        self.assertEqual(result["Config"]["Layers"]["N_sublayers"], [7])

    def test_constraints_and_missing_parameters_are_rejected(self):
        t = template()
        t["Config"]["Analysis"]["Parameter_constraints"] = ["some old constraint"]
        with self.assertRaises(ValueError):
            prepare.material_configuration(t, prepare.scan.CHEMISTRIES[0], ["x"], Path("new"))
        t = template()
        t["Config"]["Analysis"]["Parameters"] = [p for p in t["Config"]["Analysis"]["Parameters"]
                                                   if p[0] != "Erf_1_ Li_A"]
        with self.assertRaises(ValueError):
            prepare.material_configuration(t, prepare.scan.CHEMISTRIES[0], ["x"], Path("new"))


if __name__ == "__main__":
    unittest.main()
