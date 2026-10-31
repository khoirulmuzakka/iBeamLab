"""Stable public exception hierarchy for iBeamLab."""

class IBeamLabError(Exception):
    """Base exception for public iBeamLab-specific failures.

    Notes
    -----
    Catch this class for the framework exception hierarchy. Some low-level helpers
    also expose ordinary ValueError, KeyError or dependency-specific exceptions.
    """

class ValidationError(IBeamLabError, ValueError):
    """A physical description, array contract or public option is invalid.

    Notes
    -----
    Also inherits ValueError. Messages identify the invalid field when conversion
    can preserve that context.
    """

class ConfigurationError(IBeamLabError):
    """A backend or workflow configuration cannot be used.

    Notes
    -----
    Use the message to identify the configuration requiring correction.
    """

class NativeExtensionError(IBeamLabError, ImportError):
    """The compiled iBeamLab execution engine cannot be imported.

    Notes
    -----
    Also inherits ImportError. Check the installed wheel or extension search path
    and the availability of its runtime DLL/shared-library dependencies.
    """

class SimulationError(IBeamLabError):
    """Simulation execution or backend adaptation failed.

    Notes
    -----
    A backend may instead return SimulationResult.failure for an individual input;
    inspect returned failure state as well as handling exceptions.
    """

class SimnraUnavailableError(SimulationError):
    """The SIMNRA backend cannot be constructed.

    Notes
    -----
    Check that the native build enables SIMNRA and that configuration/runtime
    requirements are satisfied. Deferred COM failures may appear in result diagnostics.
    """

class DatasetError(IBeamLabError):
    """Generation, dataset publication or dataset reading failed.

    Notes
    -----
    Incomplete directories are intentionally rejected by completed-dataset readers.
    The original low-level exception is retained as the chained cause where available.
    """

class ModelPackageError(IBeamLabError):
    """A model package cannot be read, validated or written.

    Notes
    -----
    Includes unsupported development layouts, invalid metadata, checksum errors
    and unavailable paths. Package loading alone does not execute ONNX.
    """

class InferenceError(IBeamLabError):
    """Model initialization, execution or scientific processing failed.

    Notes
    -----
    Includes unsupported requested setup/isotopes, graph incompatibility, invalid
    buffers and unavailable execution providers. CUDA requests do not silently
    create CPU-only sessions.
    """
