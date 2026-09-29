"""Train the iBeamLab LRN surrogate from generated multilayer RBS datasets.

Install optional dependencies with ``pip install -e ".[training,plot]"``.
Defaults use the small 1,000-samples-per-layer-count run under examples/datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ibeamlab as ibl
from ibeamlab import LRNModel


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=ROOT / "examples" / "datasets" / "iba_multilayer_small_seed_1",
    )
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "examples" / "multilayer-generation-small.toml",
    )
    parser.add_argument("--method", default="RBS")
    parser.add_argument("--max-layers", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "examples" / "artifacts" / "lrn_rbs_5layers")
    return parser.parse_args()


def _load_training_arrays(dataset_root: Path, study: ibl.GenerationStudy,
                          method: str, maximum_layers: int) -> tuple[np.ndarray, np.ndarray]:
    expected_names = study.parameter_names
    max_spectrum_width = 0
    loaded: list[tuple[np.ndarray, np.ndarray, tuple[str, ...]]] = []

    for layer_count in range(1, maximum_layers + 1):
        path = dataset_root / f"layers_{layer_count:02d}"
        dataset = ibl.open_dataset(path)
        if not dataset.metadata.complete:
            raise RuntimeError(f"Dataset is incomplete: {path}")
        if method not in dataset.metadata.spectrum_labels:
            raise ValueError(f"{path} has no {method!r} spectrum; found {dataset.metadata.spectrum_labels}")
        if dataset.metadata.invalid or dataset.metadata.failed:
            print(f"Note: {path.name} contains invalid={dataset.metadata.invalid}, failed={dataset.metadata.failed}.")

        values = np.asarray(dataset.parameters, dtype=np.float32)
        names = dataset.metadata.parameter_names
        spectra = np.asarray(dataset.spectra(method), dtype=np.float32)
        if values.shape[0] != spectra.shape[0] or not values.shape[0]:
            raise ValueError(f"Empty or misaligned parameter/spectrum rows in {path}")
        if len(names) != values.shape[1]:
            raise ValueError(f"Parameter metadata does not match columns in {path}")
        max_spectrum_width = max(max_spectrum_width, spectra.shape[1])
        loaded.append((values, spectra, names))
        print(f"Loaded {path.name}: {values.shape[0]:,} samples, {spectra.shape[1]} channels")

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for values, spectra, names in loaded:
        name_to_column = {name: index for index, name in enumerate(names)}
        unknown = set(names) - set(expected_names)
        if unknown:
            raise ValueError(f"Dataset contains parameters outside the {maximum_layers}-layer study: {sorted(unknown)}")
        padded = np.zeros((len(values), len(expected_names)), dtype=np.float32)
        for column, name in enumerate(expected_names):
            source_column = name_to_column.get(name)
            if source_column is not None:
                padded[:, column] = values[:, source_column]
        if spectra.shape[1] < max_spectrum_width:
            spectra = np.pad(spectra, ((0, 0), (0, max_spectrum_width - spectra.shape[1])))
        features.append(padded)
        targets.append(spectra)

    x = np.concatenate(features, axis=0)
    y = np.concatenate(targets, axis=0)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Training data contains NaN or infinite values.")
    if np.any(y < 0):
        raise ValueError("RBS count targets must be nonnegative.")
    return x, y


def _make_splits(x: np.ndarray, y: np.ndarray, seed: int):
    if len(x) < 10:
        raise ValueError("At least 10 samples are needed for train/validation/test splits.")
    order = np.random.default_rng(seed).permutation(len(x))
    train_end = int(0.8 * len(order))
    validation_end = int(0.9 * len(order))
    train, validation, test = order[:train_end], order[train_end:validation_end], order[validation_end:]
    return (x[train], y[train]), (x[validation], y[validation]), (x[test], y[test])


def _scale_inputs(x: np.ndarray, parameters: tuple[ibl.Parameter, ...]):
    minimum = np.asarray([item.lower for item in parameters], dtype=np.float32)
    maximum = np.asarray([item.upper for item in parameters], dtype=np.float32)
    scale = maximum - minimum
    scale[scale <= 0] = 1.0
    return (x - minimum) / scale, minimum, scale


def _evaluate(model: LRNModel, x: np.ndarray, device: torch.device,
              batch_size: int, output_scale: float) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start:start + batch_size]).to(device)
            predictions.append(model.predict(batch).cpu().numpy())
    return np.concatenate(predictions, axis=0) / output_scale


def _evaluate_deeper_layers(dataset_root: Path, configuration: ibl.GenerationConfiguration,
                            model: LRNModel, method: str, maximum_layers: int,
                            device: torch.device, batch_size: int,
                            output_scale: float, output_dir: Path) -> None:
    """Check eager LRN extrapolation on deeper generated layer systems."""
    summaries = []
    for layer_count in range(maximum_layers + 1, configuration.maximum_layers + 1):
        path = dataset_root / f"layers_{layer_count:02d}"
        if not path.is_dir():
            continue
        dataset = ibl.open_dataset(path)
        if not dataset.metadata.complete or method not in dataset.metadata.spectrum_labels:
            print(f"Skipping unavailable generalization case: {path}")
            continue

        case_study = configuration.study(layer_count)
        case_names = case_study.parameter_names
        source_names = dataset.metadata.parameter_names
        if set(source_names) != set(case_names):
            raise ValueError(f"Unexpected parameter layout in generalization dataset {path}")
        source_index = {name: index for index, name in enumerate(source_names)}
        raw_inputs = np.asarray(dataset.parameters, dtype=np.float32)
        case_inputs = raw_inputs[:, [source_index[name] for name in case_names]]
        minimum = np.asarray([parameter.lower for parameter in case_study.parameters], dtype=np.float32)
        scale = np.asarray([parameter.upper - parameter.lower for parameter in case_study.parameters], dtype=np.float32)
        scale[scale <= 0] = 1.0
        case_inputs = (case_inputs - minimum) / scale
        case_targets = np.asarray(dataset.spectra(method), dtype=np.float32)
        predictions = _evaluate(model, case_inputs, device, batch_size, output_scale)

        common_width = min(case_targets.shape[1], predictions.shape[1])
        case_targets = case_targets[:, :common_width]
        predictions = predictions[:, :common_width]
        threshold = max(float(case_targets.max()) * 1e-3, 1.0)
        row_errors = []
        for target, prediction in zip(case_targets, predictions):
            valid = target > threshold
            if np.any(valid):
                row_errors.append(float(np.mean(np.abs(prediction[valid] - target[valid]) / target[valid])))
        median_error = float(np.median(row_errors)) if row_errors else float("nan")
        summaries.append((layer_count, len(case_inputs), median_error))
        print(f"Generalization {layer_count} layers: {len(case_inputs):,} samples, "
              f"median relative error={median_error:.5g} over bins > {threshold:.4g} counts")

        plot_count = min(4, len(case_targets))
        figure, axes = plt.subplots(plot_count, 1, figsize=(12, 2.8 * plot_count),
                                    squeeze=False, constrained_layout=True)
        for index in range(plot_count):
            axes[index, 0].plot(case_targets[index], label="SIMNRA target", linewidth=1.0)
            axes[index, 0].plot(predictions[index], label="LRN prediction", linestyle="--", linewidth=1.0)
            axes[index, 0].set_ylabel(f"sample {index}")
            axes[index, 0].grid(alpha=0.25)
            if index == 0:
                axes[index, 0].legend()
        axes[-1, 0].set_xlabel("RBS channel")
        figure.savefig(output_dir / f"generalization_{layer_count}_layers.png", dpi=160)
        plt.close(figure)
    if summaries:
        print("Deeper-layer checks use eager PyTorch variable-layer inference; the exported C++ package "
              "has the fixed training layer layout.")


def main() -> None:
    args = _arguments()
    if args.max_layers < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("max-layers, epochs, and batch-size must be positive.")
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")
    package_path = args.output_dir / f"lrn_{args.method.lower()}_{args.max_layers}layers.zip"
    if package_path.exists():
        raise FileExistsError(f"Refusing to overwrite model package: {package_path}; choose another --output-dir.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training device:", device)

    configuration = ibl.load_generation_configuration(args.config)
    if args.max_layers > configuration.maximum_layers:
        raise ValueError(f"max-layers exceeds the config limit ({configuration.maximum_layers}).")
    study = configuration.study(args.max_layers)
    x_raw, y_raw = _load_training_arrays(dataset_root, study, args.method, args.max_layers)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = _make_splits(x_raw, y_raw, args.seed)

    # The same feature-wise min-max mapping is stored in the ONNX package, so
    # C++ callers can continue passing physical parameter values.
    x_train, input_minimum, input_scale = _scale_inputs(x_train, study.parameters)
    x_val = (x_val - input_minimum) / input_scale
    x_test = (x_test - input_minimum) / input_scale
    output_scale = 1e-6
    y_train = y_train * output_scale
    y_val = y_val * output_scale

    model = LRNModel(
        study,
        {args.method: y_train.shape[1]},
        hidden_size=128,
        contribution_size=128,
        setup_embedding_dim=32,
        layer_embedding_dim=128,
        block_hidden_sizes=(256, 256),
        decoder_hidden_sizes=(512, 512),
        refiner_hidden_channels=16,
        refiner_kernel_size=17,
    ).to(device)
    print(f"Input features: {model.input_dimension}; output channels: {model.output_size}")
    print(f"Trainable parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    x_val_tensor = torch.from_numpy(x_val).to(device)
    y_val_tensor = torch.from_numpy(y_val).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    loss_function = nn.MSELoss()
    history: list[tuple[float, float]] = []
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        seen = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(inputs)
            seen += len(inputs)

        model.eval()
        with torch.inference_mode():
            validation_loss = float(loss_function(model(x_val_tensor), y_val_tensor))
        train_loss = train_loss_sum / max(seen, 1)
        history.append((train_loss, validation_loss))
        scheduler.step(validation_loss)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        tqdm.write(
            f"Epoch {epoch:03d}/{args.epochs}: train MSE={train_loss:.6g}, "
            f"validation MSE={validation_loss:.6g}, lr={optimizer.param_groups[0]['lr']:.3g}"
        )
        if stale_epochs >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")
    model.load_state_dict(best_state)
    model.to(device)

    y_prediction = _evaluate(model, x_test, device, args.batch_size, output_scale)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "lrn_state.pt"
    torch.save(
        {
            "model_state_dict": {name: value.detach().cpu()
                                 for name, value in model.state_dict().items()},
            "max_layers": args.max_layers,
            "method": args.method,
            "input_parameter_names": study.parameter_names,
            "input_minimum": input_minimum,
            "input_scale": input_scale,
            "output_scale": output_scale,
            "history": history,
        },
        weights_path,
    )
    model.export(
        package_path,
        input_minimum=input_minimum,
        input_scale=input_scale,
        output_inverse_factor=output_scale,
    )

    target_threshold = max(float(y_test.max()) * 1e-3, 1.0)
    relative_errors = []
    for target, prediction in zip(y_test, y_prediction):
        mask = target > target_threshold
        if np.any(mask):
            relative_errors.append(float(np.mean(np.abs(prediction[mask] - target[mask]) / target[mask])))
    if relative_errors:
        print(f"Test mean relative error (bins > {target_threshold:.4g} counts): "
              f"median={np.median(relative_errors):.4g}, mean={np.mean(relative_errors):.4g}")
    print("Weights:", weights_path)
    print("C++-compatible iBeamLab ONNX package:", package_path)

    plot_count = min(6, len(y_test))
    figure, axes = plt.subplots(plot_count, 1, figsize=(12, 2.8 * plot_count), squeeze=False,
                                constrained_layout=True)
    for index in range(plot_count):
        axis = axes[index, 0]
        axis.plot(y_test[index], label="SIMNRA target", linewidth=1.0)
        axis.plot(y_prediction[index], label="LRN prediction", linewidth=1.0, linestyle="--")
        axis.set_ylabel(f"sample {index} counts")
        axis.grid(alpha=0.25)
        if index == 0:
            axis.legend()
    axes[-1, 0].set_xlabel("RBS channel")
    figure_path = args.output_dir / "test_spectra.png"
    figure.savefig(figure_path, dpi=160)
    plt.show()
    print("Test spectra plot:", figure_path)

    losses = np.asarray(history, dtype=np.float64)
    history_path = args.output_dir / "training_loss.png"
    plt.figure(figsize=(8, 4))
    plt.plot(losses[:, 0], label="train")
    plt.plot(losses[:, 1], label="validation")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Scaled-count MSE")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(history_path, dpi=160)
    plt.show()
    print("Training history plot:", history_path)

    _evaluate_deeper_layers(
        dataset_root, configuration, model, args.method, args.max_layers,
        device, args.batch_size, output_scale, args.output_dir,
    )


if __name__ == "__main__":
    main()
