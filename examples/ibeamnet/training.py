"""Disk-backed minibatches, best-validation checkpointing and held-out evaluation."""
from pathlib import Path
import math
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler
from tqdm import tqdm
from .data import digest, write_json
from .model import RecurrentSurrogate, SpectrumLoss
from .validation import validate_training, validate_checkpoint_contract
from .workflow import load_cache


def scheduled_epochs(schedule, completed=0):
    """Yield remaining (zero-based epoch, one-based stage, rate) without expanding a schedule.

    The schedule describes the TOTAL run, so edits on resume affect only future
    updates. Epochs already completed retain their original recorded history.
    """
    start = 0
    for stage_number, stage in enumerate(schedule, 1):
        stop = start + stage["epochs"]
        for epoch in range(max(start, completed), stop):
            yield epoch, stage_number, stage["learning_rate"]
        start = stop


class CachedBatches(Dataset):
    """Gather whole minibatches from memory maps, avoiding per-record tensor allocation.

    The loader's sampler supplies lists of split positions. Worker processes
    reopen mappings instead of pickling the spectral arrays on Windows.
    """
    def __init__(self, cache, split):
        self.cache, self.split = str(cache), split
        self._arrays = None
        self.size = len(np.load(Path(cache) / f"{split}_indices.npy", mmap_mode="r"))

    def __len__(self):
        return self.size

    def __getstate__(self):
        return {**self.__dict__, "_arrays": None}

    def __getitem__(self, positions):
        if self._arrays is None:
            root = Path(self.cache)
            self._arrays = tuple(np.load(root / name, mmap_mode="r") for name in (
                "features.npy", "targets.npy", f"{self.split}_indices.npy"))
        features, targets, indices = self._arrays
        selected = indices[np.asarray(positions, dtype=np.int64)]
        return torch.from_numpy(features[selected]), torch.from_numpy(targets[selected])


def make_loader(cache, split, batch_size, workers, device, generator=None):
    dataset = CachedBatches(cache, split)
    sampler = RandomSampler(dataset, generator=generator) if split == "train" else SequentialSampler(dataset)
    return DataLoader(dataset, batch_size=None, sampler=BatchSampler(sampler, batch_size, False),
                      num_workers=workers, pin_memory=device.type == "cuda")


def make_model(manifest, options):
    return RecurrentSurrogate(manifest["elements"], [s["channels"] for s in manifest["spectra"]], **options)


def atomic_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") not in {"ibeamlab.forward-checkpoint", "ibeamlab.forward-weights"} or state.get("format_version") != 1:
        raise ValueError("invalid forward-surrogate checkpoint")
    if state.get("architecture") != "RecurrentSurrogate":
        raise ValueError("checkpoint is not RecurrentSurrogate; train a new model")
    return state


def evaluate(model, loader, criterion, spectra, device):
    """Stream loss and physical relative-L1 metrics without retaining predictions."""
    model.eval()
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    numerators = torch.zeros(len(spectra), device=device, dtype=torch.float64)
    denominators = torch.zeros_like(numerators)
    count = 0
    with torch.no_grad():
        for features, target in loader:
            features, target = features.to(device, non_blocking=True), target.to(device, non_blocking=True)
            predicted = model(features)
            loss = criterion(predicted, target)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite evaluation loss")
            total_loss += loss.double() * len(features)
            offset = 0
            for index, spec in enumerate(spectra):
                stop = offset + spec["channels"]
                numerators[index] += (predicted[:, offset:stop] - target[:, offset:stop]).abs().sum(dtype=torch.float64)
                denominators[index] += target[:, offset:stop].sum(dtype=torch.float64)
                offset = stop
            count += len(features)
    return {"loss": float(total_loss.cpu()) / count, "samples": count,
        "relative_l1": {s["measurement"]: float(n / d.clamp_min(1.0))
                        for s, n, d in zip(spectra, numerators.cpu(), denominators.cpu())}}


def train(config, *, resume=False, initial_weights=None):
    """Train with separate validation/test data; select weights only by validation loss.

    Resume restores optimizer, loader RNG and Torch RNG state from last.pt.
    The staged schedule is the total run and may change; other training settings,
    architecture and cache identity must match. Validation selects the best model
    across ALL stages, without early stopping or changing the declared rates.
    initial_weights starts a NEW run with matching scientific input/output
    metadata and a fresh AdamW state; source training history is not resumed.
    """
    if resume and initial_weights is not None:
        raise ValueError("choose either resume or initial_weights")
    settings, cache = config["training"], Path(config["data"]["cache"])
    total_epochs = validate_training(settings)
    schedule = settings["schedule"]
    manifest = load_cache(config)
    cache_hash = digest(cache / "manifest.json")
    output = Path(settings["output"])
    selected = settings["device"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if selected == "auto" else torch.device(selected)
    run_settings = {k: v for k, v in settings.items() if k not in {"output", "schedule"}}
    run_settings["resolved_device"] = str(device)
    if resume:
        state = load_checkpoint(output / "last.pt")
        if state["format"] != "ibeamlab.forward-checkpoint":
            raise ValueError("weights-only files require initial_weights, not resume")
        validate_checkpoint_contract(state, manifest, cache_sha256=cache_hash, model_options=config["model"],
            loss_options=config["loss"], training_settings=run_settings, context="resume checkpoint")
        if total_epochs < state["completed_epochs"]:
            raise ValueError("revised schedule is shorter than the number of completed epochs")
    else:
        state = None
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(settings["seed"])
    generator = torch.Generator().manual_seed(settings["seed"])
    model = make_model(manifest, config["model"])
    initial_digest = None
    if initial_weights is not None:
        initial = load_checkpoint(initial_weights)
        validate_checkpoint_contract(initial, manifest, model_options=config["model"],
                                     initial=True, context="initial weights")
        model.load_state_dict(initial["model_state"], strict=True)
        initial_digest = digest(Path(initial_weights))
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=schedule[0]["learning_rate"], weight_decay=settings["weight_decay"])
    criterion = SpectrumLoss(manifest["spectra"], **config["loss"])
    history, best, first_epoch = [], math.inf, 0
    if state is not None:
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        generator.set_state(state["loader_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        history, best = state["history"], state["best_validation_loss"]
        first_epoch = state["completed_epochs"]
        # A crash between publishing best.pt and last.pt must not lose a better model.
        protected = load_checkpoint(output / "best.pt")
        validate_checkpoint_contract(protected, manifest, cache_sha256=cache_hash, model_options=config["model"],
            loss_options=config["loss"], training_settings=run_settings, context="best checkpoint")
        best = min(best, protected["best_validation_loss"])
    loaders = {split: make_loader(cache, split, settings["batch_size"], settings["workers"], device, generator)
               for split in ("train", "validation", "test")}
    write_json(output / "run.json", {"format": "ibeamlab.forward-run", "cache": str(cache),
        "architecture": "RecurrentSurrogate", "ad_input_factor": manifest["ad_input_factor"],
        "cache_sha256": cache_hash, "model": config["model"], "loss": config["loss"],
        "training": {**settings, "output": str(output), "resolved_device": str(device)},
        "total_epochs": total_epochs,
        "initial_weights_sha256": initial_digest if state is None else state.get("initial_weights_sha256"),
        "split_sizes": manifest["split_sizes"]})

    def checkpoint(epoch):
        return {"format": "ibeamlab.forward-checkpoint", "format_version": 1,
            "architecture": "RecurrentSurrogate", "model_state": model.state_dict(),
            "model_options": config["model"], "loss_options": config["loss"],
            "cache_sha256": cache_hash, "epoch": epoch, "completed_epochs": epoch + 1,
            "best_validation_loss": best, "history": history,
            "schedule": schedule, "training_settings": run_settings,
            "elements": manifest["elements"], "spectra": manifest["spectra"],
            "ad_input_factor": manifest["ad_input_factor"],
            "initial_weights_sha256": initial_digest if state is None else state.get("initial_weights_sha256")}

    def save_resume(epoch):
        atomic_checkpoint(output / "last.pt", {**checkpoint(epoch),
            "optimizer_state": optimizer.state_dict(), "loader_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})

    if state is None:
        # Protect the baseline and the first epoch's starting state too.
        best = evaluate(model, loaders["validation"], criterion, manifest["spectra"], device)["loss"]
        atomic_checkpoint(output / "best.pt", checkpoint(-1))
        save_resume(-1)
        write_json(output / "history.json", history)
    print(f"Staged schedule: {total_epochs} total epochs; {first_epoch} completed; "
          f"{total_epochs - first_epoch} remaining", flush=True)
    for epoch, stage, learning_rate in scheduled_epochs(schedule, first_epoch):
        # Set AFTER restoring AdamW, so an old saved rate cannot override the stage.
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        total_loss, count = torch.zeros((), device=device, dtype=torch.float64), 0
        started, updated = time.perf_counter(), 0.
        with tqdm(loaders["train"], desc=f"Epoch {epoch + 1}/{total_epochs}, stage {stage}", unit="batch", mininterval=.5) as bar:
            for features, target in bar:
                features, target = features.to(device, non_blocking=True), target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(features), target)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                total_loss += loss.detach().double() * len(features)
                count += len(features)
                if time.perf_counter() - updated >= 1:
                    bar.set_postfix(loss=float(loss.detach().cpu()))
                    updated = time.perf_counter()
        validation = evaluate(model, loaders["validation"], criterion, manifest["spectra"], device)
        entry = {"epoch": epoch + 1, "train_loss": float(total_loss.cpu()) / count,
                 "validation_loss": validation["loss"], "learning_rate": learning_rate, "stage": stage,
                 "seconds": time.perf_counter() - started}
        history.append(entry)
        improved = validation["loss"] < best
        if improved:
            best = validation["loss"]
            atomic_checkpoint(output / "best.pt", checkpoint(epoch))
        save_resume(epoch)
        write_json(output / "history.json", history)
        print(f"Epoch {epoch + 1}, stage {stage}, lr={learning_rate:g}: "
              f"train={entry['train_loss']:.6g}, validation={validation['loss']:.6g}", flush=True)
    best_state = load_checkpoint(output / "best.pt")
    model.load_state_dict(best_state["model_state"])
    test = evaluate(model, loaders["test"], criterion, manifest["spectra"], device)
    write_json(output / "test-metrics.json", {**test, "best_epoch": best_state["epoch"] + 1,
                                           "checkpoint": "best.pt"})
    return {"history": history, "test": test, "checkpoint": str(output / "best.pt")}
