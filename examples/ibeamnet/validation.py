"""Shared, allocation-light configuration and checkpoint contract checks.

These run at workflow boundaries, never per minibatch or inference call.
Scientific model metadata remains validated by the C++ framework.
"""
from collections.abc import Mapping
from pathlib import Path
import math


def required(value, key, path):
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    if key not in value:
        raise ValueError(f"{path}.{key} is required")
    return value[key]


def number(value, path, *, minimum=0, inclusive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or (value < minimum if inclusive else value <= minimum)):
        relation = ">=" if inclusive else ">"
        raise ValueError(f"{path} must be finite and {relation} {minimum}")


def integer(value, path, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")


def seed(value, path):
    integer(value, path, minimum=0)
    if value >= 2**32:
        raise ValueError(f"{path} must be less than 2**32")


def validate_schedule(schedule):
    """Return the total run length without allocating one entry per epoch."""
    if not isinstance(schedule, (list, tuple)) or not schedule:
        raise ValueError("training.schedule must contain at least one stage")
    total = 0
    for index, stage in enumerate(schedule, 1):
        path = f"training.schedule[{index - 1}]"
        if not isinstance(stage, Mapping) or set(stage) != {"epochs", "learning_rate"}:
            raise ValueError(f"{path} requires epochs and learning_rate")
        integer(stage["epochs"], f"{path}.epochs")
        number(stage["learning_rate"], f"{path}.learning_rate")
        total += stage["epochs"]
    return total


def validate_training(training):
    """Validate settings from TOML or Python/notebook overrides identically."""
    if not isinstance(training, Mapping):
        raise ValueError("training must be a mapping")
    if {"epochs", "learning_rate", "patience"}.intersection(training):
        raise ValueError("Use training.schedule for epochs/rates; staged training has no early stopping")
    total = validate_schedule(required(training, "schedule", "training"))
    for name, minimum in (("batch_size", 1), ("workers", 0)):
        integer(required(training, name, "training"), f"training.{name}", minimum=minimum)
    number(required(training, "gradient_clip", "training"), "training.gradient_clip")
    number(required(training, "weight_decay", "training"), "training.weight_decay", inclusive=True)
    seed(required(training, "seed", "training"), "training.seed")
    integer(training.get("prefetch_factor", 4), "training.prefetch_factor")
    if type(training.get("persistent_workers", True)) is not bool:
        raise ValueError("training.persistent_workers must be boolean")
    for key, default in (("evaluation_batch_size", 2048), ("train_evaluation_samples", 20000),
                         ("validation_samples", 20000)):
        integer(training.get(key, default), f"training.{key}")
    if training.get("validation_samples", 20000) > 20000:
        raise ValueError("training.validation_samples must not exceed 20000")
    integer(training.get("test_samples", 10000), "training.test_samples")
    if training.get("test_samples", 10000) > 10000:
        raise ValueError("training.test_samples must not exceed 10000")
    integer(training.get("evaluation_workers", 0), "training.evaluation_workers", minimum=0)
    device = required(training, "device", "training")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("training.device must be a nonempty device name")
    return total


def validate_model_options(options):
    """Check architecture dimensions without importing PyTorch."""
    if not isinstance(options, Mapping):
        raise ValueError("model must be a mapping")
    for key, value in options.items():
        path = f"model.{key}"
        if key in {"block_hidden_sizes", "decoder_hidden_sizes"}:
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{path} must be a sequence of positive integers")
            for index, width in enumerate(value):
                integer(width, f"{path}[{index}]")
        elif key in {"hidden_size", "embedding_size", "contribution_size", "refiner_channels",
                     "refiner_kernel", "setup_embedding_size"}:
            integer(value, path, minimum=0 if key == "refiner_channels" else 1)
        else:
            raise ValueError(f"unknown model option: {path}")


def validate_loss(options):
    if required(options, "kind", "loss") not in ("chi_square", "mse"):
        raise ValueError("loss.kind must be chi_square or mse")
    number(required(options, "noise_floor", "loss"), "loss.noise_floor")


def validate_spectra(spectra):
    if not isinstance(spectra, (list, tuple)) or not spectra:
        raise ValueError("spectra must contain at least one detector")
    labels = set()
    for index, spec in enumerate(spectra):
        path = f"spectra[{index}]"
        label = required(spec, "measurement", path)
        if not isinstance(label, str) or not label or label in labels:
            raise ValueError(f"{path}.measurement must be a unique nonempty label")
        labels.add(label)
        if "channels" in spec:
            integer(spec["channels"], f"{path}.channels")
        for key in ("count_scale", "loss_weight"):
            number(required(spec, key, path), f"{path}.{key}")
        for key in ("exposure_scale", "rebin"):
            if type(required(spec, key, path)) is not bool:
                raise ValueError(f"{path}.{key} must be boolean")


def validate_data_config(config):
    """Validate preparation/cache settings without requiring training/export controls."""
    if required(config, "format", "config") != "ibeamlab.surrogate" or required(config, "format_version", "config") != 1:
        raise ValueError("expected ibeamlab.surrogate format_version=1")
    for section, key in (("data", "source"), ("data", "cache")):
        value = required(required(config, section, "config"), key, section)
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"{section}.{key} must be a nonempty path")
    data = required(config, "data", "config")
    layers = required(data, "layer_counts", "data")
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError("data.layer_counts must contain unique positive integers")
    for index, count in enumerate(layers):
        integer(count, f"data.layer_counts[{index}]")
    if len(set(layers)) != len(layers):
        raise ValueError("data.layer_counts must contain unique positive integers")
    for key in ("validation_fraction", "test_fraction"):
        value = required(data, key, "data")
        number(value, f"data.{key}")
        if value >= 1:
            raise ValueError(f"data.{key} must be less than 1")
    if data["validation_fraction"] + data["test_fraction"] >= 1:
        raise ValueError("validation and test fractions must leave training records")
    integer(required(data, "max_records_per_source", "data"), "data.max_records_per_source", minimum=0)
    seed(required(data, "seed", "data"), "data.seed")
    number(required(required(config, "preprocessing", "config"), "ad_input_factor", "preprocessing"),
           "preprocessing.ad_input_factor")
    validate_spectra(required(config, "spectra", "config"))


def validate_config(config):
    """Validate a complete CLI configuration; notebooks may configure phases separately."""
    validate_data_config(config)
    for section in ("training", "export"):
        value = required(required(config, section, "config"), "output", section)
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"{section}.output must be a nonempty path")
    validate_training(required(config, "training", "config"))
    validate_model_options(required(config, "model", "config"))
    validate_loss(required(config, "loss", "config"))
    export = required(config, "export", "config")
    if type(required(export, "verify_native", "export")) is not bool:
        raise ValueError("export.verify_native must be boolean")
    integer(required(export, "opset", "export"), "export.opset", minimum=17)
    validate_pileup(export.get("pileup", {}), config["spectra"])


def validate_pileup(settings, spectra):
    """Export-only detector settings, independent of training/cache contracts."""
    if not isinstance(settings, Mapping):
        raise ValueError("export.pileup must be a detector mapping")
    labels = {spec["measurement"] for spec in spectra}
    for label, values in settings.items():
        path = f"export.pileup.{label}"
        if label not in labels:
            raise ValueError(f"{path}: unknown measurement")
        if type(required(values, "enabled", path)) is not bool:
            raise ValueError(f"{path}.enabled must be boolean")
        if type(required(values, "targets_pileup_free", path)) is not bool:
            raise ValueError(f"{path}.targets_pileup_free must be boolean")
        if values["enabled"] and not values["targets_pileup_free"]:
            raise ValueError(f"{path}: pileup requires pileup-free training targets")
        number(required(values, "tau_seconds", path), f"{path}.tau_seconds", inclusive=True)


def validate_cache_contract(manifest, config):
    """Apply the existing cache checks consistently in every workflow entry point.

    This centralizes existing settings/path checks; it does not introduce new
    source hashing, sample-limit or split-fraction reuse policies.
    """
    recorded = manifest.get("spectra", [])
    configured = config["spectra"]
    if any(spec["measurement"] not in manifest.get("automatic_spectrum_lengths", [])
           for spec in configured if "channels" not in spec):
        raise ValueError("cache spectrum lengths were not inferred; prepare a new cache")
    if len(recorded) != len(configured) or any(
            {key: value for key, value in stored.items() if key != "channels" or "channels" in requested} != requested
            or not isinstance(stored.get("channels"), int) or stored["channels"] <= 0
            for stored, requested in zip(recorded, configured)):
        raise ValueError("cache.spectra differs from configuration; prepare a new cache")
    for key, expected in (("ad_input_factor", config["preprocessing"]["ad_input_factor"]),):
        if manifest.get(key) != expected:
            raise ValueError(f"cache.{key} differs from configuration; prepare a new cache")
    if sorted(s["layer_count"] for s in manifest["sources"]) != sorted(config["data"]["layer_counts"]):
        raise ValueError("cache layer counts differ from configuration; choose a new cache directory")
    if manifest["split_seed"] != config["data"]["seed"]:
        raise ValueError("cache split seed differs from configuration; choose a new cache directory")
    for source in manifest["sources"]:
        expected = Path(config["data"]["source"]) / f"layers_{source['layer_count']:02d}"
        if Path(source["path"]).resolve() != expected.resolve():
            raise ValueError("cache belongs to a different dataset root")


def validate_checkpoint_contract(state, manifest, *, cache_sha256=None, model_options=None,
                                 loss_options=None, initial=False, context="checkpoint"):
    """One compatibility check for export, resume, protected best and initial weights."""
    expected = {}
    if initial:
        expected.update((key, manifest[key]) for key in ("elements", "spectra", "ad_input_factor"))
    for key, value in (("cache_sha256", cache_sha256), ("model_options", model_options),
                       ("loss_options", loss_options)):
        if value is not None:
            expected[key] = value
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"{context}.{key} differs from the current run")
