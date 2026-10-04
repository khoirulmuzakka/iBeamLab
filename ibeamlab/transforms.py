"""Native transforms used by packaged ML models."""

from __future__ import annotations

import math
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
    thickness_bounds: tuple[float, float] | None = None,
    low: float = 0.0,
    high: float = 1.0,
) -> TransformPipeline:
    """Build concentration-normalization and parameter-bound scaling for an LRN.

    Concentrations are clipped at zero and normalized to sum to one within
    each layer.  They then pass through the scaler unchanged.  Layer thickness
    and setup parameters use their individual declared study bounds for min-max
    scaling. Optional ``thickness_bounds`` overrides the thickness scaling
    interval without changing the physical input parameters.
    """
    if high <= low:
        raise ValueError("high must be greater than low")
    parameters = tuple(study.parameters)
    if not parameters:
        raise ValueError("LRN input transform requires at least one parameter")
    if thickness_bounds is not None:
        thickness_low, thickness_high = map(float, thickness_bounds)
        if (not math.isfinite(thickness_low) or not math.isfinite(thickness_high)
                or thickness_high <= thickness_low):
            raise ValueError("thickness_bounds must be finite with high > low")
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
        elif kind in {"thickness", "layer_thickness"} and thickness_bounds is not None:
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
