"""Generate a reproducible multilayer collection from a TOML configuration."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ibeamlab as ibl

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path, nargs="?",
                        default=Path(__file__).with_name("multilayer-generation.toml"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    configuration = ibl.load_generation_configuration(args.configuration)
    configuration.validate_files()
    one_layer_total = configuration.mixed_samples + len(configuration.elements) * configuration.pure_samples_per_element
    total = one_layer_total + (configuration.maximum_layers - 1) * configuration.mixed_samples
    print(f"Configuration: {configuration.source}")
    print(f"Output: {configuration.output}")
    print(f"Elements: {', '.join(configuration.elements)}")
    print(f"Layer systems: 1..{configuration.maximum_layers}")
    print(f"One-layer samples: {one_layer_total:,}; total samples: {total:,}")
    if args.validate_only:
        print("Configuration is valid.")
        return
    summaries = configuration.run()
    for label, summary in summaries.items():
        print(label, summary)

if __name__ == "__main__":
    main()
