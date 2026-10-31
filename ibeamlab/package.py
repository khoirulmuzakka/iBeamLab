"""Read and write format 1 packages with embedded standalone ONNX graphs."""
from pathlib import Path
from ._engine import engine, to_core
from .exceptions import ModelPackageError
from .metadata import ModelMetadata


class Package:
    """Own an embedded ONNX graph and its format-1 scientific metadata.

    Parameters
    ----------
    value : ibeamlab.native.Package
        Existing native package. Prefer open() or from_onnx() in ordinary Python code.

    Notes
    -----
    Package I/O does not require ONNX execution support. Independent threads may
    open the same immutable file; each open owns its own metadata/graph storage.
    Serialize writes and metadata mutation. Loaded models own their own native
    package state; modifying this object does not reconfigure an existing session.
    """
    def __init__(self, value):
        """Wrap an owning native package; prefer Package.open or Package.from_onnx.
        """
        self._core = value

    @property
    def metadata(self):
        """Expose the package scientific contract through Python metadata objects.

        Returns
        -------
        ModelMetadata
            Directional metadata and tensor identifiers.

        Notes
        -----
        Use the setter to replace the complete contract. Nested native descriptors
        are mutable values; this is not a promise of a recursively isolated deep copy.
        Assignment/conversion does not perform complete validation until writing the
        package or constructing an inference model.
        """
        return ModelMetadata.from_core(self._core.metadata)

    @metadata.setter
    def metadata(self, value):
        """Replace the complete scientific contract.

        Parameters
        ----------
        value : ModelMetadata or native ModelMetadata
            Replacement metadata; converted to native descriptions.

        Notes
        -----
        Complete scientific validation is performed on write or model construction.
        Existing inference sessions are not reconfigured by changing a package.
        """
        self._core.metadata = to_core(value)

    def to_core(self):
        """Convert this description to an owning native value.

        Returns
        -------
        ibeamlab.native.Package
            Native value for lower-level workflows.

        Notes
        -----
        Returns the existing owning native object without copying. Mutation is visible
        through this Package wrapper; avoid concurrent mutation or writes.
        """
        return self._core

    def write(self, path):
        """Write a new format-1 package without overwriting an existing destination.

        Parameters
        ----------
        path : str or pathlib.Path
            New .ibm file path; its parent directory must exist.

        Raises
        ------
        ModelPackageError
            The destination exists, metadata is invalid, or writing fails.

        Notes
        -----
        Validates the package, writes a temporary file and publishes by rename.
        Serialize writes targeting the same path. The graph is embedded; external
        ONNX tensor files are not supported.
        """
        try:
            self._core.write(Path(path))
        except (ValueError, RuntimeError) as error:
            raise ModelPackageError(str(error)) from error

    @classmethod
    def open(cls, path):
        """Read a checksummed format-1 package and validate its metadata.

        Parameters
        ----------
        path : str or pathlib.Path
            Existing .ibm file.

        Returns
        -------
        Package
            Owning package containing graph bytes and scientific metadata.

        Raises
        ------
        ModelPackageError
            The file is missing, corrupted, unsupported or scientifically inconsistent.

        Notes
        -----
        Does not initialize an ONNX session or verify numerical graph predictions.
        Earlier development layouts lacking required geometry are rejected.
        """
        try:
            return cls(engine().Package.open(Path(path)))
        except (ValueError, RuntimeError) as error:
            raise ModelPackageError(str(error)) from error

    @classmethod
    def from_onnx(cls, path, metadata):
        """Associate a standalone ONNX file with a scientific contract.

        Parameters
        ----------
        path : str or pathlib.Path
            Existing standalone ONNX graph file.
        metadata : ModelMetadata or native ModelMetadata
            Directional scientific contract for this graph.

        Returns
        -------
        Package
            Owning in-memory package; call write() to publish a .ibm file.

        Raises
        ------
        ModelPackageError
            The graph file cannot be read or metadata is invalid.

        Notes
        -----
        Metadata is validated and default isotope assumptions are resolved explicitly.
        This operation reads graph bytes; export_package or session construction performs
        ONNX graph compatibility checks. External tensor files cannot be packaged.
        """
        try:
            return cls(engine().Package.from_onnx(Path(path), to_core(metadata)))
        except (ValueError, RuntimeError) as error:
            raise ModelPackageError(str(error)) from error
