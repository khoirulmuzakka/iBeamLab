"""Disk-backed minibatches, best-validation checkpointing and held-out evaluation."""
from pathlib import Path
import math
import random
import time
import signal
from contextlib import contextmanager
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, BatchSampler
from tqdm import tqdm
from .data import digest, write_json
from .model import RecurrentSurrogate, SpectrumLoss
from .validation import validate_training, validate_checkpoint_contract, validate_loss, validate_model_options
from .workflow import load_cache
from .splits import training_splits


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
    def __init__(self, cache, split, positions=None, split_directory=None):
        self.cache, self.split = str(cache), split
        self.split_directory = str(split_directory or cache)
        self.positions = positions
        self._arrays = None
        self.size = len(positions) if positions is not None else len(np.load(Path(self.split_directory) / f"{split}_indices.npy", mmap_mode="r"))

    def __len__(self):
        return self.size

    def __getstate__(self):
        return {**self.__dict__, "_arrays": None}

    def __getitem__(self, positions):
        if self._arrays is None:
            root = Path(self.cache)
            self._arrays = (np.load(root / "features.npy", mmap_mode="r"),
                            np.load(root / "targets.npy", mmap_mode="r"),
                            np.load(Path(self.split_directory) / f"{self.split}_indices.npy", mmap_mode="r"))
        features, targets, indices = self._arrays
        positions = np.asarray(positions, dtype=np.int64)
        selected = indices[positions if self.positions is None else self.positions[positions]]
        return torch.from_numpy(features[selected]), torch.from_numpy(targets[selected])


class EpochBatches:
    """One retained shuffle tensor; resume skips completed batches without reading them."""
    def __init__(self, size, batch_size, generator):
        self.size, self.batch_size, self.generator = size, batch_size, generator
        self.order, self.start = None, 0

    def set_epoch(self, order=None, start=0):
        if order is None:
            order = torch.randperm(self.size, generator=self.generator)
        if (order.dtype != torch.int64 or order.ndim != 1 or len(order) != self.size
                or not 0 <= start <= self.size):
            raise ValueError("invalid partial-epoch shuffle or position")
        self.order, self.start = order, start

    def __iter__(self):
        if self.order is None:
            self.set_epoch()
        for offset in range(self.start, self.size, self.batch_size):
            yield self.order[offset:offset + self.batch_size].tolist()

    def __len__(self):
        return (self.size - self.start + self.batch_size - 1) // self.batch_size


@contextmanager
def deferred_interrupt():
    """Finish the active optimizer step before handling Ctrl+C; restore the host handler."""
    requested = [False]
    previous = signal.getsignal(signal.SIGINT)
    def request(signum, frame):
        if requested[0]:
            raise KeyboardInterrupt
        requested[0] = True
    installed = False
    try:
        try:
            signal.signal(signal.SIGINT, request)
            installed = True
        except ValueError:  # A training call in a non-main thread cannot install handlers.
            pass
        yield requested
    finally:
        if installed:
            signal.signal(signal.SIGINT, previous)


def make_loader(cache, split, batch_size, workers, device, generator=None,
                prefetch_factor=4, persistent_workers=True, *, positions=None, shuffle=None, split_directory=None):
    dataset = CachedBatches(cache, split, positions, split_directory)
    if shuffle is None:
        shuffle = split == "train"
    sampler = EpochBatches(len(dataset), batch_size, generator) if shuffle else BatchSampler(
        SequentialSampler(dataset), batch_size, False)
    options = dict(prefetch_factor=prefetch_factor, persistent_workers=persistent_workers) if workers else {}
    return DataLoader(dataset, batch_size=None, sampler=sampler,
                      num_workers=workers, pin_memory=device.type == "cuda",
                      generator=torch.Generator().manual_seed(0), **options)


def evaluation_subsets(cache, manifest, settings, saved=None, split_directory=None):
    """Fixed random split positions, retained in checkpoints without copying spectra.

    A separate RNG leaves optimizer shuffling unchanged. Selected positions are
    ordered by underlying cache row to improve locality of memory-map reads.
    """
    policy = dict(train_samples=settings.get("train_evaluation_samples", 20000),
                  validation_samples=min(settings.get("validation_samples", 20000), 20000),
                  batch_size=settings.get("evaluation_batch_size", 2048), seed=settings["seed"])
    result = {"settings": policy}
    for split, limit, stream in (("train", policy["train_samples"], 100),
                                  ("validation", policy["validation_samples"], 101)):
        count = manifest["split_sizes"][split]
        size = min(count, limit)
        key = split + "_positions"
        if saved is not None and saved.get("settings") == policy:
            positions = saved[key]
            if (positions.dtype != torch.int64 or positions.ndim != 1 or len(positions) != size
                    or positions.min() < 0 or positions.max() >= count
                    or len(torch.unique(positions)) != size):
                raise ValueError(f"invalid saved {split} evaluation positions")
        else:
            rng = np.random.default_rng(np.random.SeedSequence([settings["seed"], stream]))
            chosen = rng.choice(count, size, replace=False) if size < count else np.arange(count)
            indices = np.load(Path(split_directory or cache) / f"{split}_indices.npy", mmap_mode="r")
            chosen = chosen[np.argsort(indices[chosen])]
            positions = torch.from_numpy(np.asarray(chosen, dtype=np.int64))
        result[key] = positions
    return result


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


class _StopTraining(Exception):
    pass


def evaluate(model, loader, criterion, spectra, device, *, interrupt=None):
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
            if interrupt is not None and interrupt[0]:
                raise _StopTraining
    return {"loss": float(total_loss.cpu()) / count, "samples": count,
        "relative_l1": {s["measurement"]: float(n / d.clamp_min(1.0))
                        for s, n, d in zip(spectra, numerators.cpu(), denominators.cpu())}}


def resolve_resume(output, resume):
    """Resolve True/False/'auto' without reading datasets or model weights.

    Auto starts only in a new directory. Existing directories without last.pt
    are never reused or overwritten implicitly.
    """
    output = Path(output)
    if type(resume) is not bool and resume != "auto":
        raise ValueError("resume must be True, False or 'auto'")
    checkpoint = output / "last.pt"
    if resume is True:
        if not checkpoint.is_file():
            raise ValueError(f"Cannot resume: {checkpoint} does not exist. "
                             "Use resume=False with a new run directory to start training, "
                             "or choose the directory containing your training checkpoint. "
                             "Initial weights are not a resumable training checkpoint.")
        return True
    if resume == "auto" and checkpoint.is_file():
        return True
    if output.exists():
        raise FileExistsError(f"Run directory already exists: {output}. "
                              "Resume requires last.pt; choose a new directory for a new run.")
    return False


def train(config, *, resume=False, initial_weights=None):
    """Train with separate validation/test data; select weights only by validation loss.

    Resume restores optimizer, loader RNG and Torch RNG state from last.pt.
    Training settings may change on resume; architecture and cache identity
    must match. Updated optimizer controls apply without resetting moments.
    Validation selects the best model
    across ALL stages, without early stopping or changing the declared rates.
    initial_weights starts a NEW run with matching scientific input/output
    metadata and a fresh AdamW state; source training history is not resumed.
    resume='auto' selects an existing last.pt or starts in a new directory.
    Ctrl+C in the main thread is deferred to an optimizer-step boundary. last.pt
    retains the partial epoch's shuffle and position; resume skips completed batches.
    A forced interrupt inside an operation saves a diagnostic snapshot and keeps
    the previous safe last.pt intact. Prefetch tuning can change between runs.
    """
    output = Path(config["training"]["output"])
    resume = resolve_resume(output, resume)
    if resume and initial_weights is not None:
        raise ValueError("choose either resume or initial_weights")
    if resume and not (output / "best.pt").is_file():
        raise ValueError(f"Cannot resume: {output / 'best.pt'} is missing; "
                         "exact continuation requires both best.pt and last.pt.")
    settings, cache = config["training"], Path(config["data"]["cache"])
    total_epochs = validate_training(settings)
    validate_loss(config["loss"])
    validate_model_options(config["model"])
    schedule = settings["schedule"]
    manifest = load_cache(config)
    cache_hash = digest(cache / "manifest.json")
    selected = settings["device"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if selected == "auto" else torch.device(selected)
    run_settings = {k: v for k, v in settings.items() if k not in {
        "output", "schedule", "prefetch_factor", "persistent_workers", "evaluation_batch_size",
        "train_evaluation_samples", "validation_samples", "test_samples", "evaluation_workers"}}
    run_settings["resolved_device"] = str(device)
    if resume:
        state = load_checkpoint(output / "last.pt")
        if state.get("resumable") is False or "optimizer_state" not in state:
            raise ValueError("This snapshot is not resumable; use it as initial_weights in a new run")
        if state["format"] != "ibeamlab.forward-checkpoint":
            raise ValueError("weights-only files require initial_weights, not resume")
        validate_checkpoint_contract(state, manifest, cache_sha256=cache_hash, model_options=config["model"],
                                     context="resume checkpoint")
        changed = {key: (state.get("training_settings", {}).get(key), value)
                   for key, value in run_settings.items() if state.get("training_settings", {}).get(key) != value}
        if changed:
            print(f"Applying updated training settings on resume: {changed}", flush=True)
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
    recorded_loss = state.get("best_validation_loss") if state is not None else None
    if initial_weights is not None:
        initial = load_checkpoint(initial_weights)
        validate_checkpoint_contract(initial, manifest, model_options=config["model"],
                                     initial=True, context="initial weights")
        model.load_state_dict(initial["model_state"], strict=True)
        recorded_loss = initial.get("best_validation_loss", initial.get("source", {}).get("best_validation_loss"))
        initial_digest = digest(Path(initial_weights))
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=schedule[0]["learning_rate"], weight_decay=settings["weight_decay"])
    criterion = SpectrumLoss(manifest["spectra"], **config["loss"])
    history, best, first_epoch = [], math.inf, 0
    protected = None
    training_changed = state is not None and state.get("training_settings") != run_settings
    if state is not None:
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        for group in optimizer.param_groups:
            group["weight_decay"] = settings["weight_decay"]
        if state.get("training_settings", {}).get("seed") == settings["seed"]:
            generator.set_state(state["loader_rng"])
            torch.set_rng_state(state["torch_rng"])
            if device.type == "cuda" and len(state["cuda_rng"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            if "python_rng" in state:
                random.setstate(state["python_rng"])
            if "numpy_rng" in state:
                rng = state["numpy_rng"]
                np.random.set_state((rng["kind"], rng["keys"].numpy().astype(np.uint32),
                                     rng["position"], rng["has_gauss"], rng["cached_gaussian"]))
        else:
            random.seed(settings["seed"])
            np.random.seed(settings["seed"])
            torch.manual_seed(settings["seed"])
            if device.type == "cuda":
                torch.cuda.manual_seed_all(settings["seed"])
        initial_digest = state.get("initial_weights_sha256")
        history, best = state["history"], state["best_validation_loss"]
        first_epoch = state["completed_epochs"]
        # A crash between publishing best.pt and last.pt must not lose a better model.
        protected = load_checkpoint(output / "best.pt")
        validate_checkpoint_contract(protected, manifest, cache_sha256=cache_hash, model_options=config["model"],
                                     context="best checkpoint")
        best = min(best, protected["best_validation_loss"])
    split_directory, manifest, split_policy, saved_evaluation, old_training_size, changed_validation = training_splits(
        cache, manifest, output, cache_hash, settings, state)
    split_changed = split_policy is not None and (
        state is None or state.get("split_policy") is None or old_training_size != manifest["split_sizes"]["train"])
    evaluation = evaluation_subsets(cache, manifest, settings, saved_evaluation, split_directory)
    policy = evaluation["settings"]
    evaluation_changed = state is not None and (
        changed_validation or saved_evaluation is None or state.get("evaluation", {}).get("settings") != policy
        or protected.get("evaluation", {}).get("settings") != policy
        or state.get("loss_options") != config["loss"]
        or protected.get("loss_options") != config["loss"])
    if evaluation_changed:
        print("Evaluation policy changed; rechecking the protected best on the fixed validation subset.", flush=True)
    loaders = {"train": make_loader(cache, "train", settings["batch_size"], settings["workers"], device, generator,
                                    settings.get("prefetch_factor", 4), settings.get("persistent_workers", True),
                                    split_directory=split_directory)}
    for name, split in (("train_monitor", "train"), ("validation", "validation"), ("test", "test")):
        positions = evaluation[split + "_positions"].numpy() if split != "test" else None
        loaders[name] = make_loader(cache, split, policy["batch_size"], settings.get("evaluation_workers", 0), device,
            prefetch_factor=settings.get("prefetch_factor", 4), persistent_workers=settings.get("persistent_workers", True),
            positions=positions, shuffle=False, split_directory=split_directory)
    train_monitor_count = len(evaluation["train_positions"])
    validation_count = len(evaluation["validation_positions"])
    partial = state.get("partial_epoch") if state is not None else None
    if partial and len(partial["order"]) != manifest["split_sizes"]["train"]:
        if len(partial["order"]) != old_training_size:
            raise ValueError("partial shuffle does not match the previous training pool")
        added = manifest["split_sizes"]["train"] - old_training_size
        shuffle = torch.Generator().manual_seed(manifest["split_seed"] + 203)
        partial = {**partial, "order": torch.cat((partial["order"],
            torch.randperm(added, generator=shuffle) + old_training_size))}
        print(f"Appended {added:,} returned samples to this partial epoch; completed updates are retained.", flush=True)
    if partial and partial["epoch"] != first_epoch:
        raise ValueError("partial epoch does not follow the completed checkpoint epochs")
    initial_validation = state.get("initial_validation_loss") if state is not None else None
    write_json(output / "run.json", {"format": "ibeamlab.forward-run", "cache": str(cache),
        "architecture": "RecurrentSurrogate", "ad_input_factor": manifest["ad_input_factor"],
        "cache_sha256": cache_hash, "model": config["model"], "loss": config["loss"],
        "training": {**settings, "output": str(output), "resolved_device": str(device)},
        "total_epochs": total_epochs,
        "initial_weights_sha256": initial_digest,
        "evaluation": {**policy, "train_samples": train_monitor_count, "validation_samples": validation_count},
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
            "initial_validation_loss": initial_validation,
            "evaluation": evaluation,
            "split_policy": split_policy,
            "initial_weights_sha256": initial_digest}

    def save_resume(epoch, progress=None, interrupted=False):
        rng = np.random.get_state()
        value = {**checkpoint(epoch),
            "optimizer_state": optimizer.state_dict(), "loader_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": dict(kind=rng[0], keys=torch.from_numpy(rng[1].astype(np.int64)),
                              position=rng[2], has_gauss=rng[3], cached_gaussian=rng[4]),
            "partial_epoch": progress,
            "resumable": True,
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        if interrupted:
            atomic_checkpoint(output / "interrupted.pt", value)
        atomic_checkpoint(output / "last.pt", value)

    if state is None:
        # A stop during the initial validation must also have a resumable baseline.
        atomic_checkpoint(output / "best.pt", checkpoint(-1))
        save_resume(-1)
        write_json(output / "history.json", history)

    source = output / "last.pt" if resume else initial_weights
    print(f"Loaded weights: {source if source is not None else 'random initialization'}", flush=True)
    if recorded_loss is not None:
        print(f"Checkpoint recorded best validation loss: {recorded_loss:.8g} "
              "(source metric; current loaded weights are reevaluated below)", flush=True)
    print(f"Device: {device}; batch size: {settings['batch_size']}; workers: {settings['workers']}; "
          f"prefetch: {settings.get('prefetch_factor', 4) if settings['workers'] else 'disabled (workers=0)'}", flush=True)
    print(f"Optimization samples: {manifest['split_sizes']['train']:,} (entire training pool). "
          f"Fixed evaluation: train={train_monitor_count:,}, validation={validation_count:,} "
          f"of {manifest['split_sizes']['validation']:,}; evaluation batch size={policy['batch_size']:,}. "
          f"Final test samples: {manifest['split_sizes']['test']:,}.", flush=True)
    print(f"Target count scales: { {s['measurement']: s['count_scale'] for s in manifest['spectra']} }", flush=True)
    print(f"Evaluating loaded weights BEFORE training on {validation_count:,} fixed validation samples...", flush=True)
    try:
        measured = evaluate(model, loaders["validation"], criterion, manifest["spectra"], device)["loss"]
        if evaluation_changed:
            model.load_state_dict(protected["model_state"])
            best = evaluate(model, loaders["validation"], criterion, manifest["spectra"], device)["loss"]
            model.load_state_dict(state["model_state"])
            protected.update(best_validation_loss=best, evaluation=evaluation,
                             loss_options=config["loss"], training_settings=run_settings)
            atomic_checkpoint(output / "best.pt", protected)
            initial_validation = measured
            print(f"Protected best reevaluated under the current policy: {best:.8g}", flush=True)
    except KeyboardInterrupt:
        if evaluation_changed:
            model.load_state_dict(state["model_state"])
        atomic_checkpoint(output / "interrupted.pt", {**checkpoint(first_epoch - 1),
            "interruption": "initial validation; use last.pt for exact resume", "resumable": False})
        print(f"Initial validation interrupted; saved weights to {output / 'interrupted.pt'}. "
              "The existing last.pt, if present, is unchanged.", flush=True)
        raise
    if not math.isfinite(measured):
        raise ValueError("Initial validation loss is not finite; refusing to train")
    print(f"Loaded checkpoint validation loss BEFORE training: {measured:.8g}", flush=True)
    if initial_validation is None and first_epoch == 0 and not partial:
        initial_validation = measured
    print(f"Checkpoint history last entry: {history[-1] if history else 'No completed epochs'}", flush=True)
    if partial:
        print(f"Resuming epoch {first_epoch + 1} after {partial['seen_samples']:,} completed samples; "
              "completed batches will not be read or trained again.", flush=True)
    if state is None:
        # Protect the baseline and the first epoch's starting state too.
        best = initial_validation = measured
        atomic_checkpoint(output / "best.pt", checkpoint(-1))
        save_resume(-1)
        write_json(output / "history.json", history)
    elif measured < best:
        best = measured
        atomic_checkpoint(output / "best.pt", checkpoint(first_epoch - 1))
    if evaluation_changed or training_changed or split_changed:
        save_resume(first_epoch - 1, partial)
    print(f"Protected best validation loss: {best:.8g}; checkpoint: {output / 'best.pt'}", flush=True)
    write_json(output / "startup.json", dict(weights=str(source) if source is not None else None,
        recorded_checkpoint_best_loss=recorded_loss if recorded_loss is not None and math.isfinite(recorded_loss) else None,
        loaded_weights_validation_loss=measured, protected_best_validation_loss=best,
        validation_samples=validation_count, train_monitor_samples=train_monitor_count,
        evaluation_batch_size=policy["batch_size"], completed_epochs=first_epoch,
        partial_epoch_samples=partial["seen_samples"] if partial else 0))
    state = None  # Release loaded CPU weights; optimizer and partial shuffle own what they need.
    protected = None
    print(f"Staged schedule: {total_epochs} total epochs; {first_epoch} completed; "
          f"{total_epochs - first_epoch} remaining", flush=True)
    current_epoch, count = first_epoch, 0
    sampler = loaders["train"].sampler
    try:
        with deferred_interrupt() as stop:
            for epoch, stage, learning_rate in scheduled_epochs(schedule, first_epoch):
                current_epoch = epoch
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                model.train()
                count = partial["seen_samples"] if partial else 0
                total_loss = torch.tensor(partial["loss_sum"] if partial else 0., device=device, dtype=torch.float64)
                elapsed = partial.get("elapsed_seconds", 0.) if partial else 0.
                sampler.set_epoch(partial["order"] if partial else None, count)
                partial = None
                started, updated = time.perf_counter(), 0.
                try:
                    with tqdm(total=sampler.size, initial=count,
                              desc=f"Epoch {epoch + 1}/{total_epochs}, stage {stage}", unit="samples", mininterval=.5) as bar:
                        for features, target in loaders["train"]:
                            if stop[0]:
                                raise _StopTraining
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
                            bar.update(len(features))
                            if stop[0]:
                                raise _StopTraining
                            if time.perf_counter() - updated >= 1:
                                bar.set_postfix(batch_loss=float(loss.detach().cpu()))
                                updated = time.perf_counter()
                    print(f"Epoch {epoch + 1}: evaluating fixed train={train_monitor_count:,}, "
                          f"validation={validation_count:,}, batch size={policy['batch_size']:,}...", flush=True)
                    training_metrics = evaluate(model, loaders["train_monitor"], criterion, manifest["spectra"], device, interrupt=stop)
                    validation = evaluate(model, loaders["validation"], criterion, manifest["spectra"], device, interrupt=stop)
                except _StopTraining:
                    progress = dict(epoch=epoch, stage=stage, learning_rate=learning_rate,
                        seen_samples=count, loss_sum=float(total_loss.cpu()), order=sampler.order,
                        elapsed_seconds=elapsed + time.perf_counter() - started)
                    print("Interrupt requested; saving current weights, AdamW, shuffle position and RNG states...", flush=True)
                    save_resume(epoch - 1, progress, interrupted=True)
                    write_json(output / "interruption.json", dict(epoch=epoch + 1, seen_samples=count,
                        completed_epochs=epoch, checkpoint="last.pt", resumable=True))
                    print(f"Saved {output / 'last.pt'} and {output / 'interrupted.pt'}. "
                          f"Resume continues epoch {epoch + 1} after {count:,}/{sampler.size:,} samples. "
                          f"Protected best remains {best:.8g} at {output / 'best.pt'}.", flush=True)
                    return {"history": history, "test": None, "checkpoint": str(output / "best.pt"),
                            "resume_checkpoint": str(output / "last.pt"), "interrupted": True}
                entry = {"epoch": epoch + 1, "train_loss": training_metrics["loss"],
                         "train_loss_kind": "fixed_subset", "optimization_loss": float(total_loss.cpu()) / count,
                         "validation_loss": validation["loss"], "learning_rate": learning_rate, "stage": stage,
                         "train_samples": training_metrics["samples"], "optimization_samples": count,
                         "validation_samples": validation["samples"], "evaluation_batch_size": policy["batch_size"],
                         "evaluation_settings": policy,
                         "seconds": elapsed + time.perf_counter() - started}
                history.append(entry)
                improved = validation["loss"] < best
                if improved:
                    best = validation["loss"]
                    atomic_checkpoint(output / "best.pt", checkpoint(epoch))
                save_resume(epoch)
                write_json(output / "history.json", history)
                print(f"Epoch {epoch + 1}, stage {stage}, lr={learning_rate:g}: "
                      f"train={entry['train_loss']:.6g}, validation={validation['loss']:.6g}, best={best:.6g} "
                      f"({'saved new best' if improved else 'best unchanged'}); last.pt saved", flush=True)
    except KeyboardInterrupt:
        # A forced/foreign interrupt may land inside an optimizer operation.
        # Preserve that snapshot, but do not claim it is an atomic optimizer step.
        atomic_checkpoint(output / "interrupted.pt", {**checkpoint(len(history) - 1),
            "optimizer_state": optimizer.state_dict(), "resumable": False,
            "interruption": "inside operation; exact resume uses unchanged last.pt",
            "interrupted_epoch": current_epoch + 1, "completed_samples": count})
        print(f"Forced interruption: saved current snapshot to {output / 'interrupted.pt'}. "
              f"Exact resume uses the last safe state in {output / 'last.pt'}; "
              f"best.pt is protected (validation={best:.8g}).", flush=True)
        raise
    best_state = load_checkpoint(output / "best.pt")
    model.load_state_dict(best_state["model_state"])
    test = evaluate(model, loaders["test"], criterion, manifest["spectra"], device)
    write_json(output / "test-metrics.json", {**test, "best_epoch": best_state["epoch"] + 1,
                                           "checkpoint": "best.pt"})
    return {"history": history, "test": test, "checkpoint": str(output / "best.pt"), "interrupted": False}
