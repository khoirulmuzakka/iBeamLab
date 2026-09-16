"""Native ONNX inference engine and result types."""

from ._ibeamlab_cpp.inference import (
    InferenceOptions,
    InferenceResult,
    NamedValue,
    OnnxInferenceEngine,
)

__all__ = [
    "InferenceOptions", "InferenceResult", "NamedValue", "OnnxInferenceEngine",
]
