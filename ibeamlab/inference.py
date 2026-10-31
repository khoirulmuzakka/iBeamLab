"""Per-session ONNX execution options, independent of scientific metadata."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from . import native as core
from ._engine import engine
from .exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class InferenceOptions:
    """Configure one ONNX inference session independently of model physics.

    Parameters
    ----------
    execution_provider : {"cpu", "cuda"}, default="cpu"
        Execution provider. CUDA requires a compatible GPU-enabled native build.
    device_id : int, default=0
        Nonnegative CUDA device index.
    intra_op_threads : int, default=1
        Positive thread count for parallel work within an operator.
    inter_op_threads : int, default=1
        Positive thread count configured for graph-level parallel execution.
    enable_graph_optimizations : bool, default=True
        Enable ONNX graph optimizations.
    allow_tf32 : bool, default=False
        Allow TensorFloat-32 arithmetic in supported CUDA operations.
    apply_pileup : bool, default=True
        When False, skip declared forward Pileup operations while retaining all
        other preparation. Does not add a missing operation or alter inverse inference.

    Notes
    -----
    A failed CUDA request raises an error instead of creating a CPU-only session.
    Some shape/control operators in a CUDA graph may still execute on the CPU.
    One model/session should belong to each concurrent fitting worker.
    """
    execution_provider: Literal["cpu", "cuda"] = "cpu"
    device_id: int = 0
    intra_op_threads: int = 1
    inter_op_threads: int = 1
    enable_graph_optimizations: bool = True
    allow_tf32: bool = False
    apply_pileup: bool = True

    def to_core(self) -> core.InferenceOptions:
        """Convert this description to an owning native value.

        Returns
        -------
        ibeamlab.native.InferenceOptions
            Native value for lower-level workflows.

        Notes
        -----
        Checks provider names, integer/boolean types and native numeric constraints.
        Raises ValidationError with option context when a value is invalid.
        """
        if self.execution_provider not in ("cpu", "cuda"):
            raise ValidationError("InferenceOptions.execution_provider must be 'cpu' or 'cuda'")
        for name in ("device_id", "intra_op_threads", "inter_op_threads"):
            if type(getattr(self, name)) is not int:
                raise ValidationError(f"InferenceOptions.{name} must be an integer")
        for name in ("enable_graph_optimizations", "allow_tf32", "apply_pileup"):
            if type(getattr(self, name)) is not bool:
                raise ValidationError(f"InferenceOptions.{name} must be boolean")
        bindings = engine()
        value = bindings.InferenceOptions()
        value.execution_provider = bindings.ExecutionProvider.CUDA if self.execution_provider == "cuda" else bindings.ExecutionProvider.CPU
        for name in ("device_id", "intra_op_threads", "inter_op_threads", "enable_graph_optimizations", "allow_tf32", "apply_pileup"):
            try:
                setattr(value, name, getattr(self, name))
            except (TypeError, ValueError, OverflowError) as error:
                raise ValidationError(f"InferenceOptions.{name}: {error}") from error
        try:
            value.validate()
        except (ValueError, OverflowError) as error:
            raise ValidationError(f"InferenceOptions: {error}") from error
        return value

    @classmethod
    def from_core(cls, value: core.InferenceOptions) -> InferenceOptions:
        """Construct Python-facing values from a native result.

        Parameters
        ----------
        value : ibeamlab.native.InferenceOptions
            Source value; it is not modified.

        Returns
        -------
        InferenceOptions
            Python representation of the supplied value.
        """
        return cls("cuda" if value.execution_provider == engine().ExecutionProvider.CUDA else "cpu",
                   value.device_id, value.intra_op_threads, value.inter_op_threads,
                   value.enable_graph_optimizations, value.allow_tf32, value.apply_pileup)


def available_execution_providers() -> list[str]:
    """List providers exposed by the compiled ONNX runtime.

    Returns
    -------
    list[str]
        Supported names such as "cpu" and "cuda". Empty when ONNX support is disabled.

    Raises
    ------
    NativeExtensionError
        The compiled extension cannot be imported.

    Notes
    -----
    Provider enumeration does not prove that a particular device, driver or
    model can initialize. Session construction performs the actual runtime check.
    """
    return engine().available_execution_providers()
