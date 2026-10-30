"""Compare population inference with repeated single predictions on one package."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--extension-dir", type=Path)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.population, args.layers, args.iterations) < 1:
        parser.error("population, layers and iterations must be positive")
    if args.extension_dir:
        sys.path.insert(0, str(args.extension_dir.resolve()))
    import ibeamlab as ibl
    model = ibl.load_model(args.package, options=ibl.InferenceOptions(
        execution_provider=args.provider, device_id=args.device_id))
    if not isinstance(model.metadata.model.input_preparation, ibl.LayerInputSchema):
        parser.error("package must be a forward model with layer inputs")
    elements = len(model.metadata.model.input_preparation.elements)
    ad = np.random.default_rng(42).uniform(1, 1000,
        (args.population, args.layers, elements)).astype(np.float32)
    saved = ad.copy()
    initial = model.predict_areal_density_batch(ad)
    width = sum(values.shape[1] for values in initial.values())
    out = np.empty((args.population, width), np.float64)
    expected = [model.predict_areal_density(row) for row in ad]
    relative_l1 = {}
    for label, values in initial.items():
        target = np.stack([row[label].counts for row in expected])
        error = float(np.abs(values - target).sum() / max(np.abs(target).sum(), 1e-30))
        if error > 1e-4:
            raise AssertionError(f"{label}: population/single relative L1 {error}")
        relative_l1[label] = error
    for _ in range(3):
        model.predict_areal_density_batch(ad, out=out)
    start = perf_counter()
    for _ in range(args.iterations):
        model.predict_areal_density_batch(ad, out=out)
    batch_seconds = perf_counter() - start
    # Warm the single-input shape after switching away from population tensors.
    for row in ad[:3]:
        model.predict_areal_density(row)
    start = perf_counter()
    for _ in range(args.iterations):
        for row in ad:
            model.predict_areal_density(row)
    single_seconds = perf_counter() - start
    np.testing.assert_array_equal(ad, saved)
    report = dict(provider=args.provider, device_id=args.device_id,
        population=args.population, layers=args.layers, elements=elements,
        iterations=args.iterations, processed_channels=width,
        population_ms=batch_seconds * 1000 / args.iterations,
        repeated_single_ms=single_seconds * 1000 / args.iterations,
        speedup=single_seconds / batch_seconds, relative_l1=relative_l1)
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
