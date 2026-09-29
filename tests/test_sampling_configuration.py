from pathlib import Path
import numpy as np
import ibeamlab as ibl
from ibeamlab.sampling import (
    ConcentrationRegion, ElementDistribution, ThicknessEnvelope,
    sample_layer_system, sample_thicknesses,
)

REGIONS = (
    ConcentrationRegion("trace", 0.0, 0.01),
    ConcentrationRegion("minor", 0.01, 0.1),
    ConcentrationRegion("major", 0.1, 0.5),
    ConcentrationRegion("dominant", 0.5, 1.0),
)
ELEMENTS = {
    "Li": ElementDistribution(1.0, (0.1, 0.3, 0.4, 0.2)),
    "O": ElementDistribution(1.0, (0.0, 0.1, 0.5, 0.4)),
    "Ni": ElementDistribution(0.5, (0.2, 0.3, 0.4, 0.1)),
}

def test_legacy_thickness_envelopes_are_respected():
    policy = ThicknessEnvelope(50_000, 700_000, 1.6)
    values = sample_thicknesses(1000, 4, 10, policy, np.random.default_rng(3))
    weights = 1.6 ** np.arange(4, dtype=float)
    upper = policy.maximum_for(4, 10) * weights / weights.sum()
    assert values.shape == (1000, 4)
    assert np.all(values >= 0)
    assert np.all(values <= upper + 1e-9)

def test_one_layer_adds_equal_pure_subsets_and_is_reproducible():
    kwargs = dict(layer_count=1, maximum_layers=10, mixed_count=20,
                  pure_per_element=5, master_seed=7,
                  thickness=ThicknessEnvelope(), regions=REGIONS, elements=ELEMENTS)
    first = sample_layer_system(**kwargs)
    second = sample_layer_system(**kwargs)
    assert first.thicknesses.shape == (35, 1)
    assert first.concentrations.shape == (35, 1, 3)
    assert np.array_equal(first.thicknesses, second.thicknesses)
    assert np.array_equal(first.concentrations, second.concentrations)
    assert np.allclose(first.concentrations.sum(axis=2), 1)
    assert first.report["pure_samples"] == {"Li": 5, "O": 5, "Ni": 5}

def test_example_toml_is_valid_and_rows_match_study():
    source = Path(__file__).parents[1] / "examples" / "multilayer-generation.toml"
    config = ibl.load_generation_configuration(source)
    config.validate_files()
    rows, batch = config.rows(2)
    assert rows.shape == (20_000, len(config.study(2).parameters))
    assert np.allclose(batch.concentrations.sum(axis=2), 1)
