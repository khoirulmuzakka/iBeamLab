"""Scientific model contracts and fitted processing descriptors.

TensorTransform is the union of supported fitted numerical operations.
TensorPipeline is an ordered sequence of those operations. TransformPipeline
adds direction and graph/runtime placement. SpectrumTransforms associates
independent numerical pipelines with detector labels.

SpectrumOperation is the union of physical exposure, rebin and pileup operations;
SpectrumPipeline contains per-measurement SpectrumProcessing values. These
physical operations are separate from numerical training transforms.

C++ owns complete scientific validation and execution. The Python facade adds
construction convenience and field context to conversion errors. Metadata
validation does not execute ONNX or establish model accuracy.
"""
from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias, TypeVar, Union
from . import native as core
from ._engine import engine, to_core
from .exceptions import ValidationError
from .parameters import ParameterSpec
from .sample import Isotope, Sample, ExperimentalSetup, sample_from_core, setup_from_core

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
    """Pack fixed-width physical parameter columns for a forward graph.

    Parameters
    ----------
    reference_sample : Sample or native Sample
        Reference layers and all sample properties not varied by a parameter.
    parameters : sequence of ParameterSpec or native Parameter
        Ordered physical graph columns with unique names and targets. Geometry may
        vary only when the corresponding angle targets are declared.

    Notes
    -----
    Graph shape is (batch, parameters). Unvaried setup values come from the
    ForwardModelMetadata.reference_setup. Fixed sample structure and isotope
    assumptions are checked during inference.
    """
    reference_sample: Sample | core.Sample
    parameters: Sequence[ParameterSpec | core.Parameter]

    def to_core(self) -> core.ParameterInputSchema:
        """Convert this description to an owning native value.

        Returns
        -------
        ibeamlab.native.ParameterInputSchema
            Native value for lower-level workflows.

        Notes
        -----
        Allocates native descriptions and converts nested Python values. Conversion
        does not execute a graph; validate the enclosing ModelMetadata to check the
        complete contract. Conversion errors retain field context.
        """
        return _core_value(engine().ParameterInputSchema, reference_sample=self.reference_sample,
                           parameters=self.parameters)


@dataclass(slots=True)
class ElementComposition:
    """Declare the isotope mixture assumed by one model element.

    Parameters
    ----------
    element : str
        Supported input element symbol.
    isotopes : sequence of Isotope, optional
        Fixed isotope fractions; empty resolves to the framework natural default.

    Notes
    -----
    Packages store resolved fractions explicitly. Input samples with different
    mixtures are rejected even if their elemental areal densities are identical.
    """
    element: str
    isotopes: Sequence[Isotope] = ()

    def to_core(self) -> core.ElementComposition:
        """Convert this description to an owning native value.

        Returns
        -------
        ibeamlab.native.ElementComposition
            Native value for lower-level workflows.

        Notes
        -----
        Allocates native descriptions and converts nested Python values. Conversion
        does not execute a graph; validate the enclosing ModelMetadata to check the
        complete contract. Conversion errors retain field context.
        """
        values = []
        for isotope in self.isotopes:
            raw = engine().Isotope()
            raw.massNumber, raw.fraction = isotope.mass_number, isotope.fraction
            values.append(raw)
        return _core_value(engine().ElementComposition, element=self.element, isotopes=values)


@dataclass(slots=True)
class LayerInputSchema:
    """Pack a variable-length sequence of elemental areal-density layers.

    Parameters
    ----------
    elements : sequence of str
        Unique ordered element symbols defining the last graph axis.
    isotope_compositions : sequence of ElementComposition, optional
        Per-element fixed mixtures; omitted elements use natural composition.

    Notes
    -----
    The float32 graph input has shape (batch, layers, elements). Physical features
    are thickness * atomic_fraction in 1e15 atoms/cm2, with no separate thickness
    column. Missing supported elements contribute zero. Unknown elements and
    unrepresented roughness/porosity are rejected.

    No maximum layer count is stored. The graph must actually support dynamic
    layers; metadata does not establish predictive accuracy at untested layer counts.
    Input transforms apply independently per nonempty layer, with scalar or
    per-element coefficients; zero padding remains zero.
    """
    elements: Sequence[str]
    isotope_compositions: Sequence[ElementComposition] = ()

    def to_core(self) -> core.LayerInputSchema:
        """Convert this description to an owning native value.

        Returns
        -------
        ibeamlab.native.LayerInputSchema
            Native value for lower-level workflows.

        Notes
        -----
        Allocates native descriptions and converts nested Python values. Conversion
        does not execute a graph; validate the enclosing ModelMetadata to check the
        complete contract. Conversion errors retain field context.
        """
        return _core_value(engine().LayerInputSchema, elements=self.elements,
                           isotope_compositions=[to_core(c) for c in self.isotope_compositions])


@dataclass(slots=True)
class ForwardModelMetadata:
    """Declare the scientific contract of a parameters-to-spectra model.

    Parameters
    ----------
    reference_setup : ExperimentalSetup or native ExperimentalSetup
        Reference measurements, including beam, detector geometry/calibration and exposure.
    input_preparation : ParameterInputSchema or LayerInputSchema
        Physical sample packing and isotope assumptions.
    output_spectra : SpectraSchema
        Ordered reference-condition graph outputs, including channels and pileup state.
    input_transform : TransformPipeline, optional
        Fitted numerical operations at the graph input boundary.
    output_transform : sequence of SpectrumTransform, optional
        Per-detector fitted numerical output operations, commonly applied inversely
        to restore physical counts.
    output_preparation : sequence of SpectrumProcessing, optional
        Ordered physical corrections per detector, such as exposure, rebin and pileup.

    Notes
    -----
    Execution is physical input -> input_preparation -> input_transform -> ONNX
    -> output_transform -> output_preparation. Graph-located transforms are declared
    as already embedded and are skipped at runtime. No coefficients are refitted.
    Each detector may have its own numerical and physical processing pipelines.
    """
    reference_setup: ExperimentalSetup | core.ExperimentalSetup
    input_preparation: ParameterInputSchema | LayerInputSchema | core.ParameterInputSchema | core.LayerInputSchema
    output_spectra: core.SpectraSchema
    input_transform: core.TransformPipeline = field(default_factory=lambda: engine().TransformPipeline())
    output_transform: SpectrumTransforms = field(default_factory=list)
    output_preparation: SpectrumPipeline = field(default_factory=list)

    def to_core(self) -> core.ForwardModelMetadata:
        """Convert this description to an owning native value.

        Returns
        -------
        ibeamlab.native.ForwardModelMetadata
            Native value for lower-level workflows.

        Notes
        -----
        Allocates native descriptions and converts nested Python values. Conversion
        does not execute a graph; validate the enclosing ModelMetadata to check the
        complete contract. Conversion errors retain field context.
        """
        return _core_value(engine().ForwardModelMetadata, reference_setup=self.reference_setup,
            input_preparation=self.input_preparation, output_spectra=self.output_spectra,
            input_transform=self.input_transform, output_transform=self.output_transform,
            output_preparation=self.output_preparation)


@dataclass(slots=True)
class InverseModelMetadata:
    """Declare the scientific contract of a spectra-to-depth-profile model.

    Parameters
    ----------
    reference_setup : ExperimentalSetup or native ExperimentalSetup
        Reference measurements and fixed detector geometry.
    input_spectra : SpectraSchema
        Labelled reference spectra and packed graph input layout.
    output_preparation : EdpSchema
        Element order, fixed maximum layers, physical units and uncertainty layout.
    input_preparation : sequence of SpectrumProcessing, optional
        Per-detector physical corrections from measured to reference conditions.
    input_transform : sequence of SpectrumTransform, optional
        Per-detector fitted numerical training transforms.
    output_transform : TransformPipeline, optional
        Fitted numerical output transform restoring physical elemental AD.

    Notes
    -----
    Physical input preparation runs before tensor transforms and graph execution.
    Unlike a variable-layer forward model, an inverse model has a fixed maximum
    layer count. Uncertainty packs means, presence probabilities and posterior
    standard deviations; only affine runtime output transforms can propagate
    standard deviations. Probabilities are not numerically transformed.
    """
    reference_setup: ExperimentalSetup | core.ExperimentalSetup
    input_spectra: core.SpectraSchema
    output_preparation: core.EdpSchema
    input_preparation: SpectrumPipeline = field(default_factory=list)
    input_transform: SpectrumTransforms = field(default_factory=list)
    output_transform: core.TransformPipeline = field(default_factory=lambda: engine().TransformPipeline())

    def to_core(self) -> core.InverseModelMetadata:
        """Convert this description to an owning native value.

        Returns
        -------
        ibeamlab.native.InverseModelMetadata
            Native value for lower-level workflows.

        Notes
        -----
        Allocates native descriptions and converts nested Python values. Conversion
        does not execute a graph; validate the enclosing ModelMetadata to check the
        complete contract. Conversion errors retain field context.
        """
        return _core_value(engine().InverseModelMetadata, reference_setup=self.reference_setup,
            input_spectra=self.input_spectra, output_preparation=self.output_preparation,
            input_preparation=self.input_preparation, input_transform=self.input_transform,
            output_transform=self.output_transform)


@dataclass(slots=True)
class ModelMetadata:
    """Wrap a directional scientific contract and ONNX tensor identifiers.

    Parameters
    ----------
    model : ForwardModelMetadata or InverseModelMetadata
        Directional contract; native directional descriptions are also accepted.
    input_name : str, default="inputs"
        Name of the single ONNX graph input tensor.
    output_name : str, default="outputs"
        Name of the single packed ONNX graph output tensor.
    provenance : str, default=""
        User/model provenance, for example architecture and checkpoint digest.

    Notes
    -----
    Direction is derived from model. Call validate() to check the complete
    scientific contract. This does not execute ONNX or establish predictive accuracy.
    Fitted transforms, isotope assumptions and detector geometry belong to this
    contract; runtime device/thread options belong to InferenceOptions.
    """
    model: ForwardModelMetadata | InverseModelMetadata | core.ForwardModelMetadata | core.InverseModelMetadata
    input_name: str = "inputs"
    output_name: str = "outputs"
    provenance: str = ""

    def to_core(self) -> core.ModelMetadata:
        """Convert this description to an owning native value.

        Returns
        -------
        ibeamlab.native.ModelMetadata
            Native value for lower-level workflows.

        Notes
        -----
        Allocates native descriptions and converts nested Python values. Conversion
        does not execute a graph; validate the enclosing ModelMetadata to check the
        complete contract. Conversion errors retain field context.
        """
        return _core_value(engine().ModelMetadata, model=self.model, input_name=self.input_name,
                           output_name=self.output_name, provenance=self.provenance)

    def validate(self) -> None:
        """Validate the complete directional scientific contract.

        Raises
        ------
        ValidationError
            Fields, schemas, normalized mixtures or processing declarations are inconsistent.

        Notes
        -----
        Checks dimensions, labels, geometry, units and fitted coefficients. Does not
        execute a graph, initialize an ONNX session or register custom processing kernels.
        """
        try:
            self.to_core().validate()
        except ValidationError:
            raise
        except (ValueError, TypeError, OverflowError) as error:
            raise ValidationError(f"ModelMetadata: {error}") from error

    @property
    def direction(self) -> core.Direction:
        """Return the direction derived from the directional metadata.

        Returns
        -------
        Direction
            FORWARD for sample-to-spectra, INVERSE for spectra-to-profile.
        """
        return self.to_core().direction

    def input_width(self) -> int:
        """Return the graph input feature width.

        Returns
        -------
        int
            Number of physical columns, per-layer elements, or packed spectral channels.
        """
        return self.to_core().input_width()

    def input_rank(self) -> int:
        """Return the required graph input rank.

        Returns
        -------
        int
            3 for layer sequences (batch, layers, elements); otherwise 2.
        """
        return self.to_core().input_rank()

    def layer_input(self) -> bool:
        """Report whether the input schema is a variable layer sequence.

        Returns
        -------
        bool
            True only for a forward LayerInputSchema.
        """
        return self.to_core().layer_input()

    def output_width(self) -> int:
        """Return the graph output width before runtime processing.

        Returns
        -------
        int
            Packed reference channels for forward models, or the complete EDP/uncertainty
            feature width for inverse models. Physical pileup can expand final spectra.
        """
        return self.to_core().output_width()

    @classmethod
    def from_core(cls, value: core.ModelMetadata) -> ModelMetadata:
        """Construct Python-facing values from a native result.

        Parameters
        ----------
        value : ibeamlab.native.ModelMetadata
            Source value; it is not modified.

        Returns
        -------
        ModelMetadata
            Python representation of the supplied value.

        Notes
        -----
        Creates Python directional/schema values. Nested transform and spectrum
        descriptors remain native value objects; treat loaded model metadata as read-only.
        """
        m = value.model
        if isinstance(m, core.ForwardModelMetadata):
            schema: ParameterInputSchema | LayerInputSchema
            if isinstance(m.input_preparation, core.LayerInputSchema):
                schema = LayerInputSchema(m.input_preparation.elements, [
                    ElementComposition(c.element, tuple(Isotope(i.massNumber, i.fraction) for i in c.isotopes))
                    for c in m.input_preparation.isotope_compositions])
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
