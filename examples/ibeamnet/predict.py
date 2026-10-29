"""Predict a stored dataset sample using a deployed forward package."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("prediction.npz"))
    parser.add_argument("--extension-dir", type=Path)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    if args.extension_dir:
        sys.path.insert(0, str(args.extension_dir.resolve()))
    import numpy as np
    import ibeamlab as ibl
    from ibeamlab._engine import engine
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = ibl.open_dataset(args.dataset)
    model = ibl.ForwardModel(args.package, options=ibl.InferenceOptions(
        execution_provider=args.provider, device_id=args.device_id))
    for record in dataset:
        if record.sample_index != args.sample_index:
            continue
        if record.result.failed:
            raise ValueError("requested record failed simulation")
        core = engine()
        updates = [core.ParameterUpdate(p.target, float(value))
                   for p, value in zip(dataset.metadata.parameters, record.parameters)]
        experiment = dataset.metadata.reference.to_core()
        core.apply(experiment, updates)
        result = model.predict(experiment)
        arrays = {f"prediction_{label}": spectrum.counts for label, spectrum in result.items()}
        arrays.update({f"reference_{label}": spectrum.counts for label, spectrum in record.result.items()})
        with args.output.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        print(args.output.resolve())
        return
    raise ValueError(f"sample_index {args.sample_index} was not found")


if __name__ == "__main__":
    main()
