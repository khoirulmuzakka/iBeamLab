"""Command-line preparation, training and deployment of the forward surrogate."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "export"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("surrogate.toml"))
    parser.add_argument("--extension-dir", type=Path)
    parser.add_argument("--limit", type=int, help="Maximum successful records per layer system when preparing")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initial-weights", type=Path, help="Initialize a new run from matching weights with a fresh optimizer")
    parser.add_argument("--device", help="Training device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--no-native-check", action="store_true", help="Export with Python ONNX Runtime verification only")
    args = parser.parse_args()
    if args.resume and args.initial_weights is not None:
        parser.error("--resume and --initial-weights are mutually exclusive")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    if args.extension_dir:
        sys.path.insert(0, str(args.extension_dir.resolve()))
    from examples.ibeamnet.configuration import load_config
    config = load_config(args.config)
    if args.limit is not None:
        if args.limit < 3:
            parser.error("--limit must be at least 3")
        config["data"]["max_records_per_source"] = args.limit
    if args.device:
        config["training"]["device"] = args.device
    if args.no_native_check:
        config["export"]["verify_native"] = False
    if args.command == "prepare":
        from examples.ibeamnet.data import prepare
        result = prepare(config)
        print(f"Prepared {result['records']:,} records; splits: {result['split_sizes']}")
    elif args.command == "train":
        from examples.ibeamnet.training import train
        print(train(config, resume=args.resume, initial_weights=args.initial_weights))
    else:
        from examples.ibeamnet.deployment import export
        print(export(config))


if __name__ == "__main__":
    main()
