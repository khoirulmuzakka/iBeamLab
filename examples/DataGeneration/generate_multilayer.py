"""Generate reproducible multilayer datasets using the current iBeamLab package."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.DataGeneration.configuration import load_configuration
from examples.DataGeneration.sampling import method_counts
import ibeamlab as ibl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", nargs="?", type=Path,
                        default=Path(__file__).with_name("multilayer-generation.toml"))
    parser.add_argument("--validate-only", action="store_true", help="Check settings and required reference files without simulation")
    parser.add_argument("--backend", choices=("simnra", "dummy"))
    parser.add_argument("--output", type=Path, help="Override output directory, relative to the working directory")
    parser.add_argument("--maximum-layers", type=int)
    parser.add_argument("--mixed-samples", type=int, help="Mixed samples for each layer count")
    parser.add_argument("--pure-samples", type=int, help="Additional one-layer samples for each element")
    parser.add_argument("--extension-dir", type=Path, help="CMake build/bin directory for source-tree use")
    args = parser.parse_args()
    if args.extension_dir:
        sys.path.insert(0, str(args.extension_dir.resolve()))
    try:
        configuration = load_configuration(args.configuration)
        overrides = {}
        for argument, field, minimum in ((args.maximum_layers, "maximum_layers", 1),
                (args.mixed_samples, "mixed_samples", 1), (args.pure_samples, "pure_samples", 0)):
            if argument is not None:
                if argument < minimum:
                    raise ValueError(f"{field} must be >= {minimum}")
                overrides[field] = argument
        if args.backend:
            overrides["backend"] = args.backend
        if args.output:
            overrides["output"] = args.output.resolve()
        configuration = replace(configuration, **overrides)
        configuration.validate_files()
        print(f"Configuration: {configuration.source}")
        print(f"Output: {configuration.output}")
        print(f"Elements: {', '.join(configuration.elements)}")
        print(f"Backend: {configuration.backend}; layer counts: 1..{configuration.maximum_layers}")
        if configuration.backend == "simnra":
            mode = "fast" if configuration.fast_calculation else "standard"
            print(f"SIMNRA: {configuration.workers} workers; batch_size={configuration.batch_size}; "
                  f"MTA={configuration.multithreaded_apartment}; calculation={mode}")
        counts = method_counts(configuration.mixed_samples, configuration.compound,
                               configuration.shared_random_elements, configuration.random_elements)
        print("Mixed samples per layer count: " + "; ".join(f"{count:,} {name}" for name, count in counts.items()))
        print(f"Total requested samples: {configuration.total_samples:,}")
        if args.validate_only:
            print("Configuration is valid.")
            return 0
        from tqdm import tqdm
        bar, current_layer = None, None
        measurements = len(configuration.setup.measurements)
        def progress(layer_count, completed, total):
            nonlocal bar, current_layer
            if layer_count != current_layer:
                if bar is not None:
                    bar.close()
                bar = tqdm(total=total * measurements, desc=f"{layer_count}-layer system",
                           unit="spectra", mininterval=.2, dynamic_ncols=True)
                current_layer = layer_count
            bar.update(completed * measurements - bar.n)
        try:
            status = configuration.run(layer_progress=progress)
        finally:
            if bar is not None:
                bar.close()
        for label, summary in status["datasets"].items():
            print(f"{label}: {summary['accepted']} accepted, {summary['failed']} failed, {summary['discarded']} discarded")
        return 0 if status["complete"] else 1
    except (OSError, ValueError, KeyError, TypeError, ibl.IBeamLabError) as error:
        parser.exit(1, f"Generation error: {error}\n")
    except KeyboardInterrupt:
        print("Generation interrupted; incomplete output is preserved.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
