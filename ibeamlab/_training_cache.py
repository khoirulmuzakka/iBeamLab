"""Bounded-memory building blocks for the pileup training notebook."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


def areal_density_targets(rows, case, elements, max_layers):
    """Construct one streamed batch of physical layer/element targets."""
    columns = {name: index for index, name in enumerate(case.parameter_names)}
    densities = np.zeros((len(rows), max_layers, len(elements)), dtype=np.float32)
    for layer_index, layer in enumerate(case.experiment.sample.layers):
        thickness = np.full(len(rows), layer.thickness, dtype=np.float32)
        fractions = np.tile([next(s.concentration for s in layer.species if s.element == element)
                             for element in elements], (len(rows), 1)).astype(np.float32)
        for parameter in case.parameters:
            if parameter.layer != layer_index:
                continue
            if parameter.kind == "layer_thickness":
                thickness = rows[:, columns[parameter.name]]
            elif parameter.kind == "concentration":
                fractions[:, elements.index(parameter.element)] = rows[:, columns[parameter.name]]
        if (np.any(thickness < 0) or np.any(fractions < 0)
                or not np.allclose(fractions.sum(1), 1, atol=1e-6, rtol=0)):
            raise ValueError("Invalid thickness/composition in streamed training data")
        densities[:, layer_index] = thickness[:, None] * fractions
    if not np.isfinite(densities).all():
        raise ValueError("Nonfinite training targets")
    return densities


def cached_arrays(root, signature, shapes, build, *, description):
    """Publish complete float32 .npy arrays atomically; reopen them read-only.

    A failed/interrupted build never becomes a reusable cache. Source fingerprints
    and preprocessing settings belong in signature. Existing arrays are not edited.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    destination = root / key
    if (destination / "complete.json").is_file():
        arrays = {name: np.load(destination / f"{name}.npy", mmap_mode="r") for name in shapes}
        if any(array.shape != tuple(shapes[name]) or array.dtype != np.float32
               for name, array in arrays.items()):
            raise ValueError(f"Invalid preprocessing cache: {destination}")
        print(f"Reusing {description}: {destination}")
        return arrays
    required = sum(int(np.prod(shape)) * 4 + 4096 for shape in shapes.values())
    available = shutil.disk_usage(root).free
    if required > available:
        raise OSError(f"{description} needs {required / 2**30:.2f} GiB on {root}; "
                      f"only {available / 2**30:.2f} GiB is free")
    print(f"Building {description}: {required / 2**30:.2f} GiB on disk")
    # The generated temporary directory is always inside the explicitly chosen cache root.
    with tempfile.TemporaryDirectory(prefix=f"{key[:12]}-", dir=root) as temporary:
        staging = Path(temporary)
        arrays = {}
        try:
            for name, shape in shapes.items():
                arrays[name] = np.lib.format.open_memmap(staging / f"{name}.npy", mode="w+",
                                                        dtype=np.float32, shape=shape)
            build(arrays)
            for array in arrays.values():
                array.flush()
        finally:
            for array in arrays.values():
                array._mmap.close()
        (staging / "complete.json").write_text(json.dumps(signature, sort_keys=True), encoding="utf8")
        staging.rename(destination)
    return {name: np.load(destination / f"{name}.npy", mmap_mode="r") for name in shapes}


class IndexedRows:
    """Lazy split/scale over a disk array; only requested rows are copied to RAM."""

    def __init__(self, array, indices, scale=1.0):
        self.array = array
        self.indices = np.asarray(indices, dtype=np.int64)
        self.scale = scale
        self.shape = (len(self.indices), *array.shape[1:])
        self.dtype = array.dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        values = self.array[self.indices[item]]
        return values if self.scale == 1 else values * np.float32(self.scale)

    def subset(self, indices):
        return IndexedRows(self.array, self.indices[indices], self.scale)

    def __getstate__(self):
        # NumPy's default memmap pickle copies its contents on Windows worker spawn.
        if not isinstance(self.array, np.memmap):
            raise TypeError("Worker datasets must be backed by an .npy memory map")
        return str(self.array.filename), self.indices, self.scale

    def __setstate__(self, state):
        filename, indices, scale = state
        self.__init__(np.load(filename, mmap_mode="r"), indices, scale)


class IndexedTrainingDataset:
    """Load one training sample at a time, without torch.from_numpy(full_split)."""

    def __init__(self, *arrays):
        self.arrays = arrays
        if not arrays or any(len(array) != len(arrays[0]) for array in arrays):
            raise ValueError("Training arrays must have matching row counts")

    def __len__(self):
        return len(self.arrays[0])

    def __getitem__(self, index):
        import torch
        return tuple(torch.from_numpy(np.array(array[index], dtype=np.float32, copy=True))
                     for array in self.arrays)


def log_normalization(inputs, *, batch_size=512):
    """Channelwise population moments via stable float64 batch merging."""
    mean = np.zeros(inputs.shape[1], dtype=np.float64)
    m2 = np.zeros_like(mean)
    count = 0
    with tqdm(total=len(inputs), desc="Training normalization", unit="spectra") as progress:
        for start in range(0, len(inputs), batch_size):
            values = np.log1p(inputs[start:start + batch_size]).astype(np.float64)
            batch_count = len(values)
            batch_mean = values.mean(axis=0)
            values -= batch_mean
            batch_m2 = np.einsum("ij,ij->j", values, values)
            delta = batch_mean - mean
            combined = count + batch_count
            m2 += batch_m2 + delta**2 * (count * batch_count / combined)
            mean += delta * (batch_count / combined)
            count = combined
            progress.update(batch_count)
    if not count:
        raise ValueError("Cannot normalize an empty training split")
    return mean.astype(np.float32), np.maximum(np.sqrt(m2 / count), 1e-3).astype(np.float32)


def poisson_batches(inputs, seed, output, *, batch_size=512, description="Poisson noise"):
    """Reproducible fixed draws written to the provided bounded/disk destination."""
    import torch
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with tqdm(total=len(inputs), desc=description, unit="spectra") as progress:
        for start in range(0, len(inputs), batch_size):
            stop = min(start + batch_size, len(inputs))
            expected = np.array(inputs[start:stop], dtype=np.float32, copy=True)
            output[start:stop] = torch.poisson(torch.from_numpy(expected), generator=generator).numpy()
            progress.update(stop - start)
    return output
