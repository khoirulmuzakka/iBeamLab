"""Generate a collection of fixed-schema datasets with varying layer counts.

All sampling policy lives in this Python file.  C++ receives completed parameter
rows and is responsible for SIMNRA execution and dataset storage.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate as single_layer_example
from ibeamlab.generation import (
    DataGenerator,
    FailurePolicy,
    GenerationConfig,
    GenerationOptions,
    GenerationProgress,
    LayerThickness,
)
from ibeamlab.sample import SampleModel
from ibeamlab.sampling import sample_uniform_parameters


ELEMENTS = ("Li", "Ni", "Mn", "Co", "O")


def thickness_target(layer_index: int) -> LayerThickness:
    target = LayerThickness()
    target.layer = layer_index
    return target


def thickness_envelopes(
    layer_count: int, maximum_layers: int, minimum_total: float, maximum_total: float,
    growth_ratio: float,
) -> np.ndarray:
    fraction = 1.0 if maximum_layers == 1 else (layer_count - 1) / (maximum_layers - 1)
    total = minimum_total + fraction * (maximum_total - minimum_total)
    weights = growth_ratio ** np.arange(layer_count, dtype=np.float64)
    return total * weights / weights.sum()


def build_config(layer_count: int, maximum_layers: int) -> tuple[GenerationConfig, np.ndarray]:
    common = single_layer_example.build_generation_config()
    envelopes = thickness_envelopes(layer_count, maximum_layers, 5.0e4, 7.0e5, 1.6)

    sample = SampleModel()
    sample.layers = [
        single_layer_example.layer(
            float(envelopes[index]) * 0.5,
            [(element, 1.0 / len(ELEMENTS)) for element in ELEMENTS],
        )
        for index in range(layer_count)
    ]

    config = GenerationConfig()
    config.sample = sample
    config.setup = common.setup
    config.methods = common.methods
    parameters = []
    for layer_index, upper in enumerate(envelopes):
        parameters.append(
            single_layer_example.parameter(
                f"Thickness_{layer_index + 1}",
                thickness_target(layer_index),
                0.0,
                float(upper),
                "1e15 at/cm2",
            )
        )
        parameters.extend(
            single_layer_example.parameter(
                f"Conc_{layer_index + 1}_{element}",
                single_layer_example.concentration_target(layer_index, element),
                0.0,
                1.0,
                "fraction",
            )
            for element in ELEMENTS
        )
    config.parameters = parameters
    config.validate()
    return config, envelopes


def sample_rows(
    config: GenerationConfig, envelopes: np.ndarray, sample_count: int, seed: int
) -> np.ndarray:
    rows = sample_uniform_parameters(config, sample_count, seed)
    rng = np.random.default_rng(seed)
    open_parameters = [parameter for parameter in config.parameters if parameter.fixed_value is None]
    thickness_columns = [
        index
        for index, parameter in enumerate(open_parameters)
        if isinstance(parameter.target, LayerThickness)
    ]
    # Per sample, draw a total envelope and preserve the exponential depth
    # profile. Actual layer thicknesses are independently drawn inside it.
    normalized_envelopes = envelopes / envelopes.sum()
    totals = rng.uniform(5.0e4, float(envelopes.sum()), sample_count)
    for layer_index, column in enumerate(thickness_columns):
        upper = totals * normalized_envelopes[layer_index]
        rows[:, column] = rng.uniform(0.0, upper)
    return rows


def collection_toml(maximum_layers: int, samples_per_case: int, base_seed: int) -> str:
    elements = ", ".join(f'"{element}"' for element in ELEMENTS)
    return "".join(
        [
            'format = "ibeamlab.dataset-collection"\n',
            "format_version = 1\n",
            f"minimum_layers = 1\nmaximum_layers = {maximum_layers}\n",
            f"samples_per_case = {samples_per_case}\nbase_seed = {base_seed}\n",
            f"elements = [{elements}]\n",
            'thickness_strategy = "exponential-envelope"\n',
            'concentration_strategy = "uniform-normalized-per-layer"\n',
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-layers", type=int, default=15)
    parser.add_argument("--samples-per-case", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.max_layers < 1 or args.samples_per_case < 1:
        parser.error("layer and sample counts must be positive")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_root = ROOT / "examples" / "datasets" / f"multilayer_{run_id}"
    first_config, _ = build_config(1, args.max_layers)

    with single_layer_example.build_simulator(first_config) as simulator:
        for layer_count in range(1, args.max_layers + 1):
            case_seed = args.seed + layer_count
            config, envelopes = build_config(layer_count, args.max_layers)
            rows = sample_rows(config, envelopes, args.samples_per_case, case_seed)

            options = GenerationOptions()
            options.batch_size = single_layer_example.SIMNRA_WORKERS
            options.shard_count = 1
            options.seed = case_seed
            options.sampler = "python-multilayer-exponential-envelope"
            options.sampler_version = 1
            options.sampling_config_toml = (
                'thickness_strategy = "exponential-envelope"\n'
                'concentration_strategy = "uniform-normalized-per-layer"\n'
                f"layer_count = {layer_count}\nseed = {case_seed}\n"
                f"sample_count = {args.samples_per_case}\n"
            )
            options.failure_policy = FailurePolicy.STOP

            generator = DataGenerator(config, simulator)
            output = output_root / f"layers_{layer_count:02d}"
            with tqdm(total=args.samples_per_case, unit="sample", desc=f"{layer_count} layer(s)") as bar:
                def progress(value: GenerationProgress) -> None:
                    bar.update(value.attempted - bar.n)
                    bar.set_postfix(accepted=value.accepted, failed=value.failed)

                generator.generate(output, rows.tolist(), options, progress)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "collection.toml").write_text(
        collection_toml(args.max_layers, args.samples_per_case, args.seed), encoding="utf-8"
    )
    print(f"Finished dataset collection: {output_root}")


if __name__ == "__main__":
    main()
