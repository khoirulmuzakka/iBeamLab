"""Sampling invariants and end-to-end execution of the multilayer example."""
from dataclasses import replace
from pathlib import Path
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXTENSION = None
if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
    EXTENSION = str(Path(sys.argv.pop(1)).resolve())
elif os.environ.get("IBEAMLAB_EXTENSION_DIR"):
    EXTENSION = os.environ["IBEAMLAB_EXTENSION_DIR"]
if EXTENSION:
    sys.path.insert(0, EXTENSION)
from examples.DataGeneration.configuration import load_configuration
from examples.DataGeneration.sampling import method_counts
from ibeamlab._engine import engine
from ibeamlab.exceptions import NativeExtensionError
import ibeamlab as ibl
try:
    engine()
    HAS_EXTENSION = True
except NativeExtensionError:
    HAS_EXTENSION = False

FOLDER = ROOT / "examples/DataGeneration"


class ExampleTests(unittest.TestCase):
    def configuration(self):
        config = load_configuration(FOLDER / "multilayer-generation.toml")
        return replace(config, maximum_layers=2, mixed_samples=8, pure_samples=2,
            backend="dummy", channels=8, batch_size=3,
            compound=replace(config.compound, fraction=.5),
            shared_random_elements=replace(config.shared_random_elements, fraction=.25,
                minimum_elements=2, maximum_elements=8),
            random_elements=replace(config.random_elements, fraction=.25,
                minimum_elements=1, maximum_elements=13))

    def test_sampling_reproducibility_and_compositions(self):
        config = self.configuration()
        rows, batch = config.rows(1)
        self.assertEqual(rows.shape, (34, 14))
        self.assertEqual(np.count_nonzero(np.char.startswith(np.array(batch.components), "pure_")), 26)
        np.testing.assert_allclose(batch.concentrations.sum(axis=2), 1, atol=1e-14)
        rows2, batch2 = config.rows(2)
        self.assertEqual(rows2.shape, (8, 28))
        np.testing.assert_allclose(batch2.concentrations.sum(axis=2), 1, atol=1e-14)
        np.testing.assert_array_equal(config.rows(1)[0], rows)
        np.testing.assert_array_equal(rows2[:, 0], batch2.thicknesses[:, 0])
        np.testing.assert_array_equal(rows2[:, 15:28], batch2.concentrations[:, 1, :])
        self.assertTrue(np.all(np.isfinite(rows2)))
        self.assertTrue(np.all(rows2 >= 0))
        self.assertLessEqual(batch2.thicknesses.sum(axis=1).max(), config.thickness.maximum_total)
        for i, component in enumerate(batch.components):
            if component.startswith("pure_"):
                index = tuple(config.elements).index(component.removeprefix("pure_"))
                self.assertEqual(batch.concentrations[i, 0, index], 1)
            elif component.startswith("compound_"):
                members = dict(config.compound.subsets)[component.removeprefix("compound_")]
                for j, symbol in enumerate(config.elements):
                    if symbol not in members:
                        self.assertEqual(batch.concentrations[i, 0, j], 0)

    def single_method(self, name, samples=1024):
        config = self.configuration()
        return replace(config, maximum_layers=4, mixed_samples=samples, pure_samples=0,
            compound=replace(config.compound, fraction=float(name == "compound")),
            shared_random_elements=replace(config.shared_random_elements,
                fraction=float(name == "shared_random_elements")),
            random_elements=replace(config.random_elements, fraction=float(name == "random_elements")))

    def test_shared_random_sets_and_independent_concentrations(self):
        config = self.single_method("shared_random_elements")
        _, batch = config.rows(4)
        active = batch.concentrations > 0
        np.testing.assert_array_equal(active, np.repeat(active[:, :1, :], 4, axis=1))
        self.assertEqual(set(active[:, 0].sum(axis=1)), set(range(2, 9)))
        self.assertTrue(np.any(batch.concentrations[:, 0] != batch.concentrations[:, 1]))
        self.assertEqual(set(batch.components), {"shared_random_elements"})
        self.assertEqual(batch.report["method_samples"], {
            "compound": 0, "shared_random_elements": 1024, "random_elements": 0})

    def test_layerwise_random_counts_and_sets(self):
        config = self.single_method("random_elements")
        _, batch = config.rows(4)
        active = batch.concentrations > 0
        counts = active.sum(axis=2)
        self.assertEqual(set(counts.ravel()), set(range(1, 14)))
        self.assertTrue(np.any(counts[:, 0] != counts[:, 1]))
        self.assertTrue(np.any(active[:, 0] != active[:, 1]))
        np.testing.assert_allclose(batch.concentrations.sum(axis=2), 1, atol=1e-14)
        self.assertEqual(set(batch.components), {"random_elements"})

    def test_compounds_preserve_all_configured_sets(self):
        config = self.single_method("compound", 512)
        _, batch = config.rows(4)
        subsets = dict(config.compound.subsets)
        self.assertTrue(all(batch.report["compound_samples"][name] > 0 for name in subsets))
        symbols = tuple(config.elements)
        for component, concentrations in zip(batch.components, batch.concentrations):
            members = subsets[component.removeprefix("compound_")]
            expected = np.isin(symbols, members)
            np.testing.assert_array_equal(concentrations > 0, np.repeat(expected[None, :], 4, axis=0))
        self.assertIn("Dummy_O_Na_P", subsets)
        self.assertEqual(subsets["Dummy_O_Na"], ("Na", "O"))
        self.assertEqual(subsets["Dummy_Fe_Mn_Ni_Co"], ("Fe", "Mn", "Ni", "Co"))

    def test_exact_allocation_and_disabled_compounds(self):
        config = self.configuration()
        for total in (1, 2, 7, 128):
            counts = method_counts(total, config.compound, config.shared_random_elements, config.random_elements)
            self.assertEqual(sum(counts.values()), total)
            batch = replace(config, mixed_samples=total, pure_samples=0).rows(2)[1]
            self.assertEqual(len(batch.components), total)
            np.testing.assert_allclose(batch.concentrations.sum(axis=2), 1)
        only_random = self.single_method("random_elements", 8)
        only_random = replace(only_random, compound=replace(only_random.compound, subsets=()))
        self.assertEqual(only_random.rows(2)[1].report["method_samples"]["compound"], 0)
        tied = method_counts(2, replace(config.compound, fraction=1/3),
            replace(config.shared_random_elements, fraction=1/3), replace(config.random_elements, fraction=1/3))
        self.assertEqual(tied, {"compound": 1, "shared_random_elements": 1, "random_elements": 0})

    def test_validate_only_without_extension_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "not-created"
            result = subprocess.run([sys.executable, "-I", str(FOLDER / "generate_multilayer.py"),
                "--validate-only", "--backend", "dummy", "--output", str(output)],
                check=True, capture_output=True, text=True, cwd=tmp)
            self.assertIn("Configuration is valid", result.stdout)
            self.assertFalse(output.exists())

    @unittest.skipUnless(HAS_EXTENSION, "compiled extension unavailable")
    def test_cli_shows_one_tqdm_bar_per_layer_system(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(FOLDER / "generate_multilayer.py"),
                "--backend", "dummy", "--maximum-layers", "2", "--mixed-samples", "4",
                "--pure-samples", "0", "--extension-dir", str(Path(engine().__file__).parent),
                "--output", str(Path(tmp) / "data")], check=True, capture_output=True, text=True)
            self.assertIn("1-layer system", result.stderr)
            self.assertIn("2-layer system", result.stderr)
            self.assertIn("spectra/s", result.stderr)
            self.assertNotIn("Completed samples:", result.stdout)

    def test_invalid_configuration(self):
        original = (FOLDER / "multilayer-generation.toml").read_text()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.toml"
            for old, new in [("growth_ratio = 1.6", "growth_ratio = -1"),
                    (f"fraction = {load_configuration(FOLDER / 'multilayer-generation.toml').compound.fraction:g}", "fraction = 1.5"),
                    (f"fraction = {load_configuration(FOLDER / 'multilayer-generation.toml').compound.fraction:g}", "fraction = 0.6"),
                    ("minimum_elements = 2", f"minimum_elements = {load_configuration(FOLDER / 'multilayer-generation.toml').shared_random_elements.maximum_elements + 1}"),
                    ("minimum_elements = 1", "minimum_elements = 0"),
                    ("maximum_elements = 13", "maximum_elements = 14"),
                    ('count_distribution = "uniform"', 'count_distribution = "unsupported"'),
                    ('selection = "uniform_without_replacement"', 'selection = "with_replacement"'),
                    ('symbol = "Li"', 'symbol = "Li"\nselection_probability = 0.5'),
                    ('particle = "H"\n', 'particle = "He"\n'),
                    ('elements = ["Li", "C"]', 'elements = ["Unconfigured", "C"]')]:
                # Change one occurrence so the particle mismatch is intentional.
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1))
                with self.subTest(setting=old, replacement=new):
                    with self.assertRaises(ValueError):
                        load_configuration(path)

    @unittest.skipUnless(HAS_EXTENSION, "compiled extension unavailable")
    def test_generation_and_audit_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(self.configuration(), output=Path(tmp) / "data", records_per_shard=3)
            progress = []
            layer_progress = []
            status = config.run(lambda done, total: progress.append((done, total)),
                layer_progress=lambda layers, done, total: layer_progress.append((layers, done, total)))
            self.assertTrue(status["complete"])
            self.assertEqual(status["accepted"], 42)
            self.assertEqual(progress[-1], (42, 42))
            self.assertEqual([p[0] for p in progress], list(range(1, 43)))
            self.assertEqual(layer_progress[0], (1, 0, 34))
            self.assertIn((1, 34, 34), layer_progress)
            self.assertIn((2, 0, 8), layer_progress)
            self.assertEqual(layer_progress[-1], (2, 8, 8))
            self.assertTrue(json.loads((config.output / "generation-summary.json").read_text())["complete"])
            resolved = json.loads((config.output / "generation.resolved.json").read_text())
            self.assertEqual(resolved["backend"], "dummy")
            self.assertEqual(resolved["mixed_samples_per_layer_count"], 8)
            self.assertEqual(resolved["method_samples_per_layer_count"], {
                "compound": 4, "shared_random_elements": 2, "random_elements": 2})
            self.assertEqual(len(resolved["configuration_sha256"]), 64)
            for layers, count in [(1, 34), (2, 8)]:
                path = config.output / f"layers_{layers:02d}"
                dataset = ibl.open_dataset(path)
                self.assertEqual(len(dataset), count)
                np.testing.assert_array_equal(dataset.parameters, config.rows(layers)[0])
                self.assertEqual(dataset.spectra("RBS").shape, (count, 8))
                with (path / "sampling-components.csv").open(newline="") as stream:
                    audited = list(csv.DictReader(stream))
                    self.assertEqual(len(audited), count)
                    batch = config.rows(layers)[1]
                    for index, row in enumerate(audited):
                        members = json.loads(row["layer_elements"])
                        counts = json.loads(row["layer_element_counts"])
                        self.assertEqual(counts, [len(m) for m in members])
                        expected = [[symbol for symbol, c in zip(config.elements, layer) if c > 0]
                                    for layer in batch.concentrations[index]]
                        self.assertEqual(members, expected)
                audit = json.loads((path / "sampling-summary.json").read_text())
                self.assertEqual(sum(audit["components"].values()), count)
            with self.assertRaises(FileExistsError):
                config.run()

    @unittest.skipUnless(HAS_EXTENSION, "compiled extension unavailable")
    def test_interrupted_generation_preserves_incomplete_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(self.configuration(), output=Path(tmp) / "interrupted")
            def interrupt(done, total):
                if done == 2:
                    raise RuntimeError("example interruption")
            with self.assertRaises(ibl.DatasetError):
                config.run(interrupt)
            status = json.loads((config.output / "generation-summary.json").read_text())
            self.assertFalse(status["complete"])
            self.assertEqual(status["active_dataset"], "layers_01")
            self.assertIn("example interruption", status["error"])
            self.assertFalse((config.output / "layers_02").exists())
            with self.assertRaises(ibl.DatasetError):
                ibl.open_dataset(config.output / "layers_01")


if __name__ == "__main__":
    unittest.main()
