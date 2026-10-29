"""Optional training utilities for user-supplied PyTorch models.

Arrays must already be in training units with preprocessing applied. Export
the fitted preprocessing descriptors alongside the graph for physical inference.
PyTorch is imported only when fit() is called.
"""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    """Sample-weighted mean loss per epoch and optional validation loss."""
    loss: tuple[float, ...]
    validation_loss: tuple[float, ...]


def fit(model, inputs, targets, *, epochs=100, batch_size=64, learning_rate=1e-3,
        validation_data=None, optimizer=None, loss_fn=None, seed=42):
    """Train a CPU model on preprocessed float32 rows, preserving its original mode.

    Defaults to Adam and mean-squared error. Supply a scalar mean-reduced loss
    for other objectives. Shuffling uses an independent seeded generator.
    """
    import torch
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")

    def tensors(x, y):
        x, y = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
            raise ValueError("training arrays must be nonempty aligned 2D matrices")
        if x.device.type != "cpu" or y.device.type != "cpu":
            raise ValueError("fit requires CPU arrays and a CPU model")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("training arrays must be finite")
        return x, y

    x, y = tensors(inputs, targets)
    validation = tensors(*validation_data) if validation_data is not None else None
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("fit requires a CPU model")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) if optimizer is None else optimizer
    loss_fn = torch.nn.MSELoss() if loss_fn is None else loss_fn
    generator = torch.Generator().manual_seed(seed)
    losses, validation_losses = [], []
    was_training = model.training
    try:
        for _ in range(epochs):
            model.train()
            total = 0.0
            order = torch.randperm(len(x), generator=generator)
            for indices in order.split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                predicted = model(x[indices])
                if predicted.shape != y[indices].shape:
                    raise ValueError("model output shape differs from training targets")
                loss = loss_fn(predicted, y[indices])
                if loss.ndim or not torch.isfinite(loss):
                    raise ValueError("loss must be a finite scalar")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            losses.append(total / len(x))
            if validation is not None:
                model.eval()
                with torch.no_grad():
                    prediction = model(validation[0])
                    if prediction.shape != validation[1].shape:
                        raise ValueError("model output shape differs from validation targets")
                    loss = loss_fn(prediction, validation[1])
                    if loss.ndim or not torch.isfinite(loss):
                        raise ValueError("validation loss must be a finite scalar")
                    validation_losses.append(float(loss))
    finally:
        model.train(was_training)
    return TrainingHistory(tuple(losses), tuple(validation_losses))
