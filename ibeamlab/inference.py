"""Per-session ONNX execution options, independent of scientific metadata."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from . import native as core
from ._engine import engine
from .exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class InferenceOptions:
    """Choose CPU/CUDA, device and threading for one inference session.

    CUDA requires a GPU-enabled native build and compatible device libraries.
    A failed CUDA request raises an error instead of selecting a CPU-only session.
    CPU shape/control operators can still participate in a CUDA graph.
    apply_pileup=False skips declared forward Pileup steps, retaining all other
    processing. It never changes the package, adds a missing step or affects inverse inference.
    """
    execution_provider: Literal["cpu", "cuda"] = "cpu"
    device_id: int = 0
    intra_op_threads: int = 1
    inter_op_threads: int = 1
    enable_graph_optimizations: bool = True
    allow_tf32: bool = False
    apply_pileup: bool = True

    def to_core(self) -> core.InferenceOptions:
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
        return cls("cuda" if value.execution_provider == engine().ExecutionProvider.CUDA else "cpu",
                   value.device_id, value.intra_op_threads, value.inter_op_threads,
                   value.enable_graph_optimizations, value.allow_tf32, value.apply_pileup)


def available_execution_providers() -> list[str]:
    """Return providers supported by the native build/runtime; availability needs working drivers too."""
    return engine().available_execution_providers()
