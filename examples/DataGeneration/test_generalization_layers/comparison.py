"""Streaming surrogate evaluation through the public iBeamLab batch inference API."""
from itertools import islice
from pathlib import Path
import csv
import hashlib
import json
import time

import numpy as np
import ibeamlab as ibl
from ibeamlab._engine import engine


def chi_square(observed, expected, predicted, *, minimum_expected=1.):
    """Sum (data-prediction)^2/(1+data) on a fixed expected-count ROI.

    No parameters are fitted: normalized scores divide by evaluated bins, not
    bins minus a fitted parameter count. Noise-only scores use observed data in
    the denominator too; their low-count expectation need not equal one.
    """
    observed, expected, predicted = map(np.asarray, (observed, expected, predicted))
    if observed.shape != expected.shape or predicted.shape != expected.shape or expected.ndim != 1:
        raise ValueError("spectra must have identical one-dimensional shapes")
    if not np.isfinite(minimum_expected) or minimum_expected <= 0:
        raise ValueError("ROI threshold must be positive and finite")
    for values in (observed, expected, predicted):
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("counts must be finite and nonnegative")
    mask = expected >= minimum_expected
    bins = int(mask.sum())
    if not bins:
        raise ValueError("no channels meet the expected-count ROI threshold")
    y, mu, prediction = observed[mask], expected[mask], predicted[mask]
    value = float(np.sum((y - prediction) ** 2 / (1. + y)))
    oracle = float(np.sum((y - mu) ** 2 / (1. + y)))
    discrepancy = float(np.sum((prediction - mu) ** 2 / (1. + mu)))
    return dict(evaluated_bins=bins, chi2=value, chi2_per_bin=value / bins,
                noise_chi2_per_bin=oracle / bins, model_error_per_bin=discrepancy / bins)


def case_information(root, layers):
    """Scan stored channel lengths in native code without loading a spectral matrix."""
    native = engine()
    result = {}
    for count in layers:
        path = Path(root) / f"layers_{count:02d}"
        expected, noisy = native.DatasetReader(path / "expected"), native.DatasetReader(path / "noisy")
        if expected.metadata.accepted != noisy.metadata.accepted or expected.metadata.failed or noisy.metadata.failed:
            raise ValueError(f"incomplete or failed records in {path}")
        if len(expected.metadata.reference.sample.layers) != count:
            raise ValueError(f"layer count mismatch in {path}")
        result[count] = dict(path=path, samples=expected.metadata.accepted,
                             maximum_channels=noisy.maximum_spectrum_lengths()["RBS"])
    return result


def load_surrogate(package_path, cases, options):
    """Adapt only the evaluation grid in an in-memory package snapshot.

    A shared bare-grid capacity covers every measured full-pileup spectrum.
    Native inference rebins independently to each sample's calibration, applies
    its exposure and pileup, then comparison uses only recorded channel support.
    ZeroOutside explicitly allows requested edges beyond the reference grid.
    The package file, weights, numerical transforms and pileup constant are unchanged.
    """
    path = Path(package_path)
    if not path.is_file():
        raise FileNotFoundError(f"Export the trained surrogate first or set MODEL_PATH: {path}")
    package = ibl.Package.open(path)
    metadata = package.metadata
    forward = metadata.model
    if not isinstance(forward, ibl.ForwardModelMetadata) or not isinstance(forward.input_preparation, ibl.LayerInputSchema):
        raise ValueError("comparison requires a variable-layer forward surrogate")
    if [s.measurement for s in forward.output_spectra.spectra] != ["RBS"]:
        raise ValueError("this notebook evaluates a single RBS output")
    if forward.output_spectra.spectra[0].state != ibl.SpectrumState.BARE:
        raise ValueError("surrogate reference output must be pileup-free")
    pipeline = next((p for p in forward.output_preparation if p.measurement == "RBS"), None)
    if pipeline is None or [type(s).__name__ for s in pipeline.steps] != ["Rebin", "ExposureScale", "Pileup"]:
        raise ValueError("export RBS output_preparation as Rebin -> ExposureScale -> Pileup")
    bare_channels = max((item["maximum_channels"] + 2) // 2 for item in cases.values())
    original = pipeline.steps
    pipeline.steps = [ibl.Rebin(bare_channels, require_full_coverage=False), *original[1:]]
    forward.output_preparation = [pipeline]
    metadata.model = forward
    package.metadata = metadata
    model = ibl.ForwardModel(package, options=options)
    return model, dict(package=str(path.resolve()), package_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        bare_channels=bare_channels, processed_channels=2 * bare_channels - 1,
        rebin_coverage="zero_outside_reference", comparison_support="recorded SIMNRA channels only",
        pileup_tau_seconds=original[-1].tau_seconds)


def record_pairs(path):
    native = engine()
    a, b = native.DatasetReader(path / "expected"), native.DatasetReader(path / "noisy")
    for x, y in zip(a, b, strict=True):
        if x.sample_index != y.sample_index or x.sample_id != y.sample_id or x.parameters != y.parameters:
            raise ValueError("expected/noisy row identity or parameter mismatch")
        yield x, y


def reconstruct(reference, parameters, values):
    """Reconstruct the full sample and its actual calibration/exposure together."""
    native = engine()
    experiment = native.Experiment()
    experiment.sample, experiment.setup = reference.sample, reference.setup
    native.apply(experiment, [native.ParameterUpdate(p.target, float(v)) for p, v in zip(parameters, values, strict=True)])
    return experiment


def predict_pairs(model, path, pairs):
    """One ONNX call for this batch, with independently processed measurements."""
    metadata = engine().DatasetReader(path / "expected").metadata
    experiments = [reconstruct(metadata.reference, metadata.parameters, x.parameters) for x, _ in pairs]
    predictions = model.predict_batch(experiments)
    for (x, y), experiment, prediction in zip(pairs, experiments, predictions, strict=True):
        expected = np.asarray(x.result.spectra[0].counts_array)
        observed = np.asarray(y.result.spectra[0].counts_array)
        full_prediction = prediction["RBS"].counts
        if len(full_prediction) < len(expected):
            raise ValueError("evaluation grid does not cover recorded spectrum")
        measurement = experiment.setup.measurements[0]
        yield x, expected, observed, full_prediction[:len(expected)], measurement


def evaluate(model, cases, output, *, batch_size=128, limit_per_case=None,
             minimum_expected=1., progress=None, provenance=None):
    """Write scalar metrics incrementally; keep one inference batch in memory.

    CSV is opened exclusively, so existing results are never silently reused or
    overwritten. Interrupted runs retain the partial CSV and complete=false.
    """
    if batch_size < 1 or (limit_per_case is not None and limit_per_case < 1):
        raise ValueError("batch size and optional sample limit must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    status = dict(complete=False, batch_size=batch_size, limit_per_case=limit_per_case,
                  minimum_expected=minimum_expected,
                  chi2_definition="sum((data-prediction)^2/(1+data))",
                  inference_options=dict(execution_provider=model.options.execution_provider,
                                         device_id=model.options.device_id),
                  **(provenance or {}))
    report = output / "comparison.json"
    start = time.perf_counter()
    fields = ["layers", "sample_index", "sample_id", "channels", "evaluated_bins", "chi2", "chi2_per_bin",
              "noise_chi2_per_bin", "model_error_per_bin", "particles_sr", "calibration_linear", "calibration_offset"]
    with (output / "scores.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        report.write_text(json.dumps(status, indent=2) + "\n")
        try:
            for layers, case in cases.items():
                count = min(case["samples"], limit_per_case or case["samples"])
                pairs = islice(record_pairs(case["path"]), count)
                done = 0
                while batch := list(islice(pairs, batch_size)):
                    for record, expected, observed, predicted, measurement in predict_pairs(model, case["path"], batch):
                        score = chi_square(observed, expected, predicted,
                            minimum_expected=minimum_expected)
                        writer.writerow(dict(layers=layers, sample_index=record.sample_index,
                            sample_id=record.sample_id, channels=len(expected), **score,
                            particles_sr=measurement.acquisition.particlesSr,
                            calibration_linear=measurement.detector.calibrationLinear,
                            calibration_offset=measurement.detector.calibrationOffset))
                    done += len(batch)
                    if progress:
                        progress(layers, done, count)
                stream.flush()
            status["complete"] = True
        finally:
            status["elapsed_seconds"] = time.perf_counter() - start
            report.write_text(json.dumps(status, indent=2) + "\n")
    return output / "scores.csv"


def select_examples(model, cases, frame, *, quantiles=(0, .1, .25, .5, .75, .9, 1)):
    """Seven score quantiles per layer system, chosen after evaluating all rows."""
    result = {}
    for layers, case in cases.items():
        rows = frame[frame.layers == layers].sort_values("chi2_per_bin")
        chosen = [rows.iloc[round(q * (len(rows) - 1))] for q in quantiles]
        indices = {int(row.sample_index) for row in chosen}
        pairs = []
        for pair in record_pairs(case["path"]):
            if pair[0].sample_index in indices:
                pairs.append(pair)
                if len(pairs) == len(indices):
                    break
        predictions = {record.sample_index: (expected.copy(), observed.copy(), predicted.copy(), measurement)
                       for record, expected, observed, predicted, measurement in predict_pairs(model, case["path"], pairs)}
        result[layers] = [(q, row, predictions[int(row.sample_index)]) for q, row in zip(quantiles, chosen, strict=True)]
    return result
