"""Generate pileup-enabled RBS generalization cases with independent Poisson observations."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_case(base, layers, samples, workers, seed):
    """Keep the original composition mixture and total thickness cap for every case."""
    import numpy as np
    import ibeamlab as ibl
    reference = ROOT / "xnra/Ref_RBS_LiCOFNaAlSiPSTiMnFeCoNiCuH.xnra"
    measurement = next(m for m in base.setup.measurements if m.label == "RBS")
    configuration = replace(base, maximum_layers=layers, mixed_samples=samples,
        pure_samples=0, seed=seed, workers=workers, backend="simnra",
        setup=ibl.ExperimentalSetup([measurement]),
        methods=(ibl.SimnraMethod("RBS", reference, measurement.beam.particle),))
    configuration.validate_files()
    flags = [(n.text or "").strip().lower() for n in ET.parse(reference).getroot().iter()
             if n.tag.split("}")[-1].lower() == "pileup"]
    if flags != ["true"]:
        raise ValueError("SIMNRA reference must enable pileup")
    plan = configuration.plan(layers)
    rows, batch = configuration.rows(layers)
    exposure = measurement.acquisition.particles_sr
    linear = measurement.detector.calibration_linear
    additional = (
        ibl.vary.particles_sr("RBS", bounds=(.9 * exposure, 1.1 * exposure)),
        ibl.vary.calibration_linear("RBS", bounds=(.95 * linear, 1.05 * linear)),
        ibl.vary.calibration_offset("RBS", bounds=(-10, 10)),
    )
    rng = np.random.default_rng(np.random.SeedSequence([seed, layers, 2001]))
    setup_rows = np.column_stack([rng.uniform(*p.bounds, samples) for p in additional])
    rows = np.ascontiguousarray(np.column_stack((rows, setup_rows)))
    return configuration, ibl.GenerationPlan(plan.reference, (*plan.parameters, *additional)), rows, batch


def add_poisson(expected, noisy, *, seed, layers, shard_size):
    """Stream one shard at a time; preserve row identity and all physical parameters."""
    import numpy as np
    from ibeamlab._engine import engine
    native = engine()
    reader = native.DatasetReader(expected)
    metadata = reader.metadata
    total = metadata.accepted
    metadata.accepted = metadata.failed = metadata.discarded = 0
    metadata.provenance += f"; observations=Poisson(expected_counts); noise_seed={seed},{layers},2002"
    writer = native.DatasetWriter(noisy, metadata, shard_size)
    rng = np.random.default_rng(np.random.SeedSequence([seed, layers, 2002]))
    from tqdm import tqdm
    for record in tqdm(reader, total=total, desc=f"{layers}-layer Poisson", unit="spectra"):
        result = record.result
        spectra = []
        for spectrum in result.spectra:
            counts = np.asarray(spectrum.counts_array)
            if not np.all(np.isfinite(counts)) or np.any(counts < 0):
                raise ValueError(f"invalid expected counts in sample {record.sample_index}")
            spectra.append(native.Spectrum(spectrum.measurement, rng.poisson(counts).astype(np.float64)))
        result.spectra = spectra
        record.result = result
        writer.append(record)
    writer.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-dir", type=Path, default=ROOT / "build/local/Release/bin")
    parser.add_argument("--configuration", type=Path, default=HERE.parent / "multilayer-generation.toml")
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--layers", type=int, nargs="+", default=[10, 13, 16, 20])
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if (args.samples < 1 or not 1 <= args.workers <= 256 or any(v < 1 for v in args.layers)
            or len(set(args.layers)) != len(args.layers) or args.seed < 0):
        parser.error("samples/layers must be positive and layer counts unique; seed >= 0; workers in 1..256")
    sys.path.insert(0, str(args.extension_dir.resolve()))
    import ibeamlab as ibl
    from ibeamlab._engine import engine
    from examples.DataGeneration.configuration import load_configuration, _hash
    from tqdm import tqdm
    base = load_configuration(args.configuration)
    if args.validate_only:
        for layers in args.layers:
            make_case(base, layers, args.samples, args.workers, args.seed)
            print(f"Validated {layers} layers, {args.samples:,} samples; pileup reference enabled")
        return
    for layers in args.layers:
        if (args.output / f"layers_{layers:02d}").exists():
            raise FileExistsError(f"case already exists: {args.output / f'layers_{layers:02d}'}")
    args.output.mkdir(parents=True, exist_ok=True)
    status = dict(complete=False, samples_per_case=args.samples, layers=args.layers, cases={})
    summary_path = args.output / "generalization-summary.json"
    write_json(summary_path, status)
    try:
        # One persistent pool; references are loaded once per worker.
        simulator = None
        try:
            for layers in args.layers:
                cfg, plan, rows, batch = make_case(base, layers, args.samples, args.workers, args.seed)
                folder = args.output / f"layers_{layers:02d}"
                expected, noisy = folder / "expected", folder / "noisy"
                print(f"{layers} layers: {args.samples:,} RBS spectra; {args.workers} workers; "
                      f"pileup=true; MTA={cfg.multithreaded_apartment}; fast={cfg.fast_calculation}", flush=True)
                # Refuse existing case directories so partial outputs are never overwritten.
                folder.mkdir(exist_ok=False)
                resolved = cfg.resolved()
                resolved.update(output=str(folder.resolve()), layer_count=layers, total_samples=args.samples,
                    noise="Poisson after SIMNRA pileup", noise_stream=[args.seed, layers, 2002],
                    varied_parameters=[dict(name=p.name, bounds=p.bounds, unit=p.unit)
                                       for p in plan.parameters[-3:]],
                    base_configuration_sha256=_hash(base.source))
                write_json(folder / "generation.resolved.json", resolved)
                (folder / "base-generation.toml").write_bytes(base.source.read_bytes())
                write_json(folder / "sampling-summary.json", batch.report)
                if simulator is None:
                    simulator = ibl.SimnraSimulator(cfg.methods, workers=args.workers,
                        multithreaded_apartment=cfg.multithreaded_apartment,
                        fast_calculation=cfg.fast_calculation)
                with tqdm(total=args.samples, desc=f"{layers}-layer SIMNRA", unit="spectra",
                          mininterval=1) as bar:
                    def progress(done, total):
                        bar.update(done - bar.n)
                    result = plan.generate(rows, simulator, expected, progress=progress,
                        batch_size=cfg.batch_size, records_per_shard=cfg.records_per_shard,
                        provenance=f"generalization; layers={layers}; seed={args.seed}; pileup=true")
                if result.cancelled or result.accepted != args.samples or result.failed:
                    raise RuntimeError(f"incomplete {layers}-layer simulation: {result}")
                status["cases"][str(layers)] = dict(expected=str(expected), accepted=result.accepted,
                                                   complete=False)
                write_json(summary_path, status)
                add_poisson(expected, noisy, seed=args.seed, layers=layers, shard_size=cfg.records_per_shard)
                observed = engine().DatasetReader(noisy).metadata
                if observed.accepted != args.samples:
                    raise RuntimeError("noisy dataset count mismatch")
                status["cases"][str(layers)].update(noisy=str(noisy), complete=True)
                write_json(summary_path, status)
                del rows, batch
        finally:
            if simulator is not None:
                simulator.close()
        status["complete"] = True
    except BaseException as error:
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(summary_path, status)


if __name__ == "__main__":
    main()
