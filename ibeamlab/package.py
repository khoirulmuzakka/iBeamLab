"""Read and write format 1 packages with embedded standalone ONNX graphs."""
from pathlib import Path
from ._engine import engine, to_core
from .exceptions import ModelPackageError
from .metadata import ModelMetadata


class Package:
    """An owning C++ package with Python-facing scientific metadata."""
    def __init__(self, value):
        self._core = value

    @property
    def metadata(self):
        """A metadata snapshot; assign it back to update the package."""
        return ModelMetadata.from_core(self._core.metadata)

    @metadata.setter
    def metadata(self, value):
        self._core.metadata = to_core(value)

    def to_core(self):
        return self._core

    def write(self, path):
        """Write format 1, refusing to overwrite an existing file."""
        try:
            self._core.write(Path(path))
        except (ValueError, RuntimeError) as error:
            raise ModelPackageError(str(error)) from error

    @classmethod
    def open(cls, path):
        try:
            return cls(engine().Package.open(Path(path)))
        except (ValueError, RuntimeError) as error:
            raise ModelPackageError(str(error)) from error

    @classmethod
    def from_onnx(cls, path, metadata):
        try:
            return cls(engine().Package.from_onnx(Path(path), to_core(metadata)))
        except (ValueError, RuntimeError) as error:
            raise ModelPackageError(str(error)) from error
