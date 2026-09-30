"""Native transforms used by packaged ML models."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generation import GenerationStudy

from ._ibeamlab_cpp.transforms import (
    ConstantFactorTransform,
    IdentityTransform,
    LayerwiseConcentrationNormalizer,
    LogTransform,
    MinMaxScaler,
    ParameterBoundMinMaxScaler,
    StandardScaler,
    Transform,
    TransformPipeline,
)


def build_lrn_input_transform(
    study: "GenerationStudy",
    *,
    thickness_bounds: tuple[float, float] = (0.0, 100_000.0),
    low: float = 0.0,
    high: float = 1.0,
) -> TransformPipeline:
    """Build concentration-normalization and parameter-bound scaling for an LRN.

    Concentrations are clipped at zero and normalized to sum to one within
    each layer.  They then pass through the scaler unchanged.  Layer thickness
    uses ``thickness_bounds``; all other parameters use their declared study
    bounds.
    """
    if high <= low:
        raise ValueError("high must be greater than low")
    thickness_low, thickness_high = map(float, thickness_bounds)
    if thickness_high <= thickness_low:
        raise ValueError("thickness_bounds must satisfy high > low")

    parameters = tuple(study.parameters)
    if not parameters:
        raise ValueError("LRN input transform requires at least one parameter")
    concentration_groups: dict[int, list[int]] = defaultdict(list)
    minimum: list[float] = []
    scale: list[float] = []
    for index, parameter in enumerate(parameters):
        kind = str(parameter.kind).strip().lower().replace("-", "_").replace(" ", "_")
        if kind == "concentration":
            concentration_groups[int(parameter.layer or 0)].append(index)
            # Normalized concentrations already occupy the target interval.
            minimum.append(float(low))
            scale.append(float(high - low))
        elif kind in {"thickness", "layer_thickness"}:
            minimum.append(thickness_low)
            scale.append(thickness_high - thickness_low)
        else:
            bound_low = float(parameter.lower)
            bound_high = float(parameter.upper)
            minimum.append(bound_low)
            scale.append(bound_high - bound_low if bound_high > bound_low else 1.0)

    normalizer = LayerwiseConcentrationNormalizer(
        len(parameters),
        [concentration_groups[layer] for layer in sorted(concentration_groups)],
    )
    scaler = ParameterBoundMinMaxScaler(minimum, scale, low, high)
    pipeline = TransformPipeline()
    pipeline.add(normalizer)
    pipeline.add(scaler)
    return pipeline

__all__ = [
    "ConstantFactorTransform",
    "IdentityTransform",
    "LayerwiseConcentrationNormalizer",
    "LogTransform",
    "MinMaxScaler",
    "ParameterBoundMinMaxScaler",
    "StandardScaler",
    "Transform",
    "TransformPipeline",
    "build_lrn_input_transform",
]
