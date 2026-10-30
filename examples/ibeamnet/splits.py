"""Bound held-out pools and return every surplus record to optimizer training."""
from hashlib import sha256
import json
from pathlib import Path
import numpy as np


def _select(pool, limit, seed, stream, preferred=None):
    size = min(len(pool), limit)
    if preferred is not None and len(preferred) == size:
        chosen = np.asarray(preferred, dtype=np.int64)
    elif size == len(pool):
        chosen = np.arange(size)
    else:
        chosen = np.random.default_rng(np.random.SeedSequence([seed, stream])).choice(len(pool), size, replace=False)
    keep = np.zeros(len(pool), dtype=bool)
    keep[chosen] = True
    return np.asarray(pool[chosen], dtype=np.int64), np.asarray(pool[~keep], dtype=np.int64)


def cap_split_indices(splits, seed, validation_max=20000, test_max=10000):
    """Preserve the original training prefix; append surplus held-out rows."""
    validation, extra_validation = _select(splits["validation"], validation_max, seed, 201)
    test, extra_test = _select(splits["test"], test_max, seed, 202)
    return dict(train=np.concatenate((splits["train"], extra_validation, extra_test)),
                validation=validation, test=test)


def _write_indices(path, parts):
    temporary = path.with_suffix(".tmp.npy")
    values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.int64,
                                      shape=(sum(len(part) for part in parts),))
    offset = 0
    for part in parts:
        values[offset:offset + len(part)] = part
        offset += len(part)
    values.flush()
    del values
    temporary.replace(path)


def training_splits(cache, manifest, output, cache_hash, settings, state=None):
    """Reuse spectral cache; publish only small run-local index files.

    Previous holdouts remain held out or move into training, never the reverse.
    Retain the old optimizer order as a prefix so a partial epoch can append the
    newly available samples without replaying completed optimizer updates.
    """
    import torch
    previous = state.get("split_policy") if state else None
    saved_evaluation = state.get("evaluation") if state else None
    validation_max = min(settings.get("validation_samples", 20000), 20000)
    test_max = min(settings.get("test_samples", 10000), 10000)
    if previous is None and (manifest["split_sizes"]["validation"] <= validation_max
                             and manifest["split_sizes"]["test"] <= test_max):
        return Path(cache), manifest, None, saved_evaluation, manifest["split_sizes"]["train"], False
    cache, output = Path(cache), Path(output)
    if previous is not None:
        if previous["cache_sha256"] != cache_hash:
            raise ValueError("run split indices belong to a different cache")
        validation_pool = previous["validation_indices"].numpy()
        test_pool = previous["test_indices"].numpy()
        additional = previous["additional_training_indices"].numpy()
    else:
        validation_pool = np.load(cache / "validation_indices.npy", mmap_mode="r")
        test_pool = np.load(cache / "test_indices.npy", mmap_mode="r")
        additional = np.empty(0, dtype=np.int64)
    old_training_size = manifest["split_sizes"]["train"] + len(additional)
    preferred = saved_evaluation.get("validation_positions") if saved_evaluation else None
    if preferred is not None:
        preferred = preferred.numpy()
    validation, extra_validation = _select(validation_pool, validation_max, manifest["split_seed"], 201, preferred)
    test, extra_test = _select(test_pool, test_max, manifest["split_seed"], 202)
    extras = np.concatenate((additional, extra_validation, extra_test))
    policy = dict(cache_sha256=cache_hash, validation_indices=torch.from_numpy(validation),
                  test_indices=torch.from_numpy(test), additional_training_indices=torch.from_numpy(extras))
    digest = sha256(cache_hash.encode())
    for values in (validation, test, extras):
        digest.update(values.tobytes())
    selection_id = digest.hexdigest()
    directory = output / "splits"
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "manifest.json"
    counts = dict(train=manifest["split_sizes"]["train"] + len(extras), validation=len(validation), test=len(test))
    cached = json.loads(marker.read_text()) if marker.is_file() else {}
    if cached.get("selection_id") != selection_id or any(
            not (directory / f"{name}_indices.npy").is_file() for name in counts):
        original_training = np.load(cache / "train_indices.npy", mmap_mode="r")
        _write_indices(directory / "train_indices.npy", (original_training, extras))
        _write_indices(directory / "validation_indices.npy", (validation,))
        _write_indices(directory / "test_indices.npy", (test,))
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(selection_id=selection_id, split_sizes=counts), indent=2))
        temporary.replace(marker)
    if sum(counts.values()) != manifest["records"]:
        raise ValueError("run split counts do not cover every cached record")
    preserved_validation = (preferred is not None and len(preferred) == len(validation))
    if saved_evaluation is not None:
        saved_evaluation = dict(saved_evaluation)
        if preserved_validation:
            saved_evaluation["validation_positions"] = torch.arange(len(validation), dtype=torch.int64)
        else:
            saved_evaluation = None
        if saved_evaluation is not None and len(saved_evaluation["train_positions"]) != min(
                counts["train"], settings.get("train_evaluation_samples", 20000)):
            saved_evaluation = None
    print(f"Effective split: train={counts['train']:,}, validation={counts['validation']:,}, "
          f"test={counts['test']:,}; no records discarded. Spectral cache unchanged.", flush=True)
    changed_validation = previous is None and not preserved_validation
    return directory, {**manifest, "split_sizes": counts}, policy, saved_evaluation, old_training_size, changed_validation
