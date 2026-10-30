"""Typed construction of the format-1 scientific model contract.

Descriptors are native value types. C++ owns scientific validation and processing;
Python adds field context to conversion errors without duplicating those rules.
"""
from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias, TypeVar, Union
from . import native as core
from ._engine import engine, to_core
from .exceptions import ValidationError
from .parameters import ParameterSpec
from .sample import Sample, ExperimentalSetup, sample_from_core, setup_from_core

TensorTransform: TypeAlias = Union["core.Log1p", "core.MinMaxScale", "core.StandardScale", "core.MultiplyScale", "core.CustomTransform"]
TensorPipeline: TypeAlias = Sequence[TensorTransform]
SpectrumOperation: TypeAlias = Union["core.ExposureScale", "core.Rebin", "core.Pileup", "core.RemovePileup"]
SpectrumPipeline: TypeAlias = Sequence["core.SpectrumProcessing"]
SpectrumTransforms: TypeAlias = Sequence["core.SpectrumTransform"]
T = TypeVar("T")


def _core_value(factory: Callable[[], T], **fields: Any) -> T:
    """Convert once at a metadata boundary and retain the failing field's path."""
    value = factory()
    for name, item in fields.items():
        try:
            converted = [to_core(p) for p in item] if name == "parameters" else to_core(item)
            setattr(value, name, converted)
        except ValidationError:
            raise
        except (TypeError, ValueError, OverflowError, AttributeError) as error:
            raise ValidationError(f"{factory.__name__}.{name}: {error}") from error
    return value


@dataclass(slots=True)
class ParameterInputSchema:
    """Fixed physical columns and unvaried sample properties for parameter models."""
    reference_sample: Sample | core.Sample
    parameters: Sequence[ParameterSpec | core.Parameter]

    def to_core(self) -> core.ParameterInputSchema:
        return _core_value(engine().ParameterInputSchema, reference_sample=self.reference_sample,
                           parameters=self.parameters)


@dataclass(slots=True)
class LayerInputSchema:
    """Unbounded layer sequence of physical elemental AD (1e15 atoms/cm2).

    Graph input is [batch,layers,elements]. Preprocessing applies per nonempty
    layer; coefficients are scalar or per element. Empty padding stays zero.
    """
    elements: Sequence[str]

    def to_core(self) -> core.LayerInputSchema:
        return _core_value(engine().LayerInputSchema, elements=self.elements)


@dataclass(slots=True)
class ForwardModelMetadata:
    """Sample packing, training transforms, and independent physical detector corrections.

    input_preparation declares physical parameters or elemental AD packing.
    input_transform applies numerical training transforms; output_transform
    restores physical reference spectra before output_preparation maps them to
    requested measurement conditions. Embedded graph transforms execute only in ONNX.
    """
    reference_setup: ExperimentalSetup | core.ExperimentalSetup
    input_preparation: ParameterInputSchema | LayerInputSchema | core.ParameterInputSchema | core.LayerInputSchema
    output_spectra: core.SpectraSchema
    input_transform: core.TransformPipeline = field(default_factory=lambda: engine().TransformPipeline())
    output_transform: SpectrumTransforms = field(default_factory=list)
    output_preparation: SpectrumPipeline = field(default_factory=list)

    def to_core(self) -> core.ForwardModelMetadata:
        return _core_value(engine().ForwardModelMetadata, reference_setup=self.reference_setup,
            input_preparation=self.input_preparation, output_spectra=self.output_spectra,
            input_transform=self.input_transform, output_transform=self.output_transform,
            output_preparation=self.output_preparation)


@dataclass(slots=True)
class InverseModelMetadata:
    """Measured spectra to fixed-maximum-layer physical EDPs.

    Physical input_preparation precedes per-detector numerical input_transform.
    output_transform restores densities; affine transforms scale posterior std
    by absolute scale and leave probabilities untouched. output_preparation
    declares layer/element layout and physical units. Nonlinear uncertainty
    transforms are rejected rather than approximated.
    """
    reference_setup: ExperimentalSetup | core.ExperimentalSetup
    input_spectra: core.SpectraSchema
    output_preparation: core.EdpSchema
    input_preparation: SpectrumPipeline = field(default_factory=list)
    input_transform: SpectrumTransforms = field(default_factory=list)
    output_transform: core.TransformPipeline = field(default_factory=lambda: engine().TransformPipeline())

    def to_core(self) -> core.InverseModelMetadata:
        return _core_value(engine().InverseModelMetadata, reference_setup=self.reference_setup,
            input_spectra=self.input_spectra, output_preparation=self.output_preparation,
            input_preparation=self.input_preparation, input_transform=self.input_transform,
            output_transform=self.output_transform)


@dataclass(slots=True)
class ModelMetadata:
    """Directional contract, graph tensor names and user provenance."""
    model: ForwardModelMetadata | InverseModelMetadata | core.ForwardModelMetadata | core.InverseModelMetadata
    input_name: str = "inputs"
    output_name: str = "outputs"
    provenance: str = ""

    def to_core(self) -> core.ModelMetadata:
        return _core_value(engine().ModelMetadata, model=self.model, input_name=self.input_name,
                           output_name=self.output_name, provenance=self.provenance)

    def validate(self) -> None:
        try:
            self.to_core().validate()
        except ValidationError:
            raise
        except (ValueError, TypeError, OverflowError) as error:
            raise ValidationError(f"ModelMetadata: {error}") from error

    @property
    def direction(self) -> core.Direction:
        return self.to_core().direction

    def input_width(self) -> int:
        return self.to_core().input_width()

    def input_rank(self) -> int:
        return self.to_core().input_rank()

    def layer_input(self) -> bool:
        return self.to_core().layer_input()

    def output_width(self) -> int:
        return self.to_core().output_width()

    @classmethod
    def from_core(cls, value: core.ModelMetadata) -> ModelMetadata:
        m = value.model
        if isinstance(m, core.ForwardModelMetadata):
            schema: ParameterInputSchema | LayerInputSchema
            if isinstance(m.input_preparation, core.LayerInputSchema):
                schema = LayerInputSchema(m.input_preparation.elements)
            else:
                schema = ParameterInputSchema(sample_from_core(m.input_preparation.reference_sample),
                    [ParameterSpec(p.name, p.target, p.unit) for p in m.input_preparation.parameters])
            directional: ForwardModelMetadata | InverseModelMetadata = ForwardModelMetadata(
                setup_from_core(m.reference_setup), schema, m.output_spectra,
                m.input_transform, m.output_transform, m.output_preparation)
        else:
            directional = InverseModelMetadata(setup_from_core(m.reference_setup),
                m.input_spectra, m.output_preparation, m.input_preparation,
                m.input_transform, m.output_transform)
        return cls(directional, value.input_name, value.output_name, value.provenance)
