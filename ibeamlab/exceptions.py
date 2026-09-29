"""Stable public exception hierarchy for iBeamLab."""

class IBeamLabError(Exception):
    """Base class for all public iBeamLab errors."""

class ValidationError(IBeamLabError, ValueError):
    """A scientific domain object or option is invalid."""

class ConfigurationError(IBeamLabError):
    """A backend or workflow is incorrectly configured."""

class NativeExtensionError(IBeamLabError, ImportError):
    """The compiled execution engine cannot be loaded."""

class SimulationError(IBeamLabError):
    """A spectrum simulation failed."""

class SimnraUnavailableError(SimulationError):
    """The optional SIMNRA installation is unavailable."""

class DatasetError(IBeamLabError):
    """A dataset cannot be generated or read."""

class ModelPackageError(IBeamLabError):
    """A packaged model is invalid or unavailable."""

class InferenceError(IBeamLabError):
    """Model inference failed."""
