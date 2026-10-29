"""Measure real multilayer generation throughput, including dataset writing.

Run separately from other SIMNRA tests. Uses temporary output, never production
datasets. Reports cold warmup separately from steady-state generation.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import sys
import tempfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= 256 or args.samples < 1:
        parser.error("workers must be in [1,256] and samples must be positive")
    sys.path.insert(0, str(args.extension_dir.resolve()))
    import ibeamlab as ibl
    from examples.DataGeneration.configuration import load_configuration
    config = replace(load_configuration(ROOT / "examples/DataGeneration/multilayer-generation.toml"),
                     mixed_samples=max(8, args.samples, args.workers * 2), pure_samples=0)
    if not 1 <= args.layers <= config.maximum_layers:
        parser.error(f"layers must be between 1 and {config.maximum_layers}")
    plan = config.plan(args.layers)
    rows, _ = config.rows(args.layers)
    reports = []
    with tempfile.TemporaryDirectory(prefix="ibeamlab-simnra-throughput-") as temporary:
        directory = Path(temporary)
        for name, workers, batch_size, fast, samples in [
                ("serial_standard", 1, 1, False, 8),
                ("parallel_fast", args.workers, 512, True, args.samples)]:
            print(f"Running {name}: {workers} workers, batch_size={batch_size}", flush=True)
            with ibl.SimnraSimulator(config.methods, workers=workers, fast_calculation=fast) as simulator:
                warmup = min(len(rows), workers * 2)
                started = time.perf_counter()
                plan.generate(rows[:warmup], simulator, directory / f"{name}-warmup", batch_size=batch_size)
                warmup_seconds = time.perf_counter() - started
                started = time.perf_counter()
                result = plan.generate(rows[:samples], simulator, directory / name,
                                       batch_size=batch_size, records_per_shard=256)
                seconds = time.perf_counter() - started
                if fast:
                    def interrupt(done, total):
                        if done == 1:
                            raise RuntimeError("throughput cancellation probe")
                    interrupted = directory / "interrupted"
                    try:
                        plan.generate(rows, simulator, interrupted, progress=interrupt,
                                      batch_size=min(len(rows), 4096))
                    except ibl.DatasetError as error:
                        if "throughput cancellation probe" not in str(error):
                            raise
                    else:
                        raise AssertionError("progress interruption was ignored")
                    try:
                        ibl.open_dataset(interrupted)
                    except ibl.DatasetError:
                        pass
                    else:
                        raise AssertionError("interrupted dataset was published")
                    simulator.reset_stop()
                    if simulator.simulate(plan.reference).failed:
                        raise AssertionError("backend failed after draining interrupted jobs")
            dataset = ibl.open_dataset(result.path)
            np.testing.assert_array_equal(dataset.parameters, rows[:samples])
            if result.accepted != samples or result.failed or result.cancelled:
                raise RuntimeError("benchmark generation failed")
            record = {"mode": name, "workers": workers, "batch_size": batch_size, "layers": args.layers,
                "samples": samples, "fast_calculation": fast,
                "warmup_seconds": warmup_seconds, "generation_seconds": seconds,
                "spectra_per_second": samples * len(config.setup.measurements) / seconds}
            reports.append(record)
            print(json.dumps(record), flush=True)
    if args.report:
        args.report.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
