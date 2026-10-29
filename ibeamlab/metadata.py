"""Python construction of the format 1 scientific model contract.

Processing descriptors are the C++ value types exported from ibeamlab. Their
ordered lists are serialized by the core; the wrapper does not duplicate those
algorithms. Spectrum processing is keyed by measurement label.
"""
from dataclasses import dataclass, field
from ._engine import engine, to_core
from .parameters import ParameterSpec
from .sample import experiment_from_core, setup_from_core


@dataclass(slots=True)
class ForwardModelMetadata:
    """Parameters to packed labelled spectra, with independent output pipelines."""
    reference: object
    input_parameters: list
    output_spectra: object
    preprocessing: list = field(default_factory=list)
    postprocessing: list = field(default_factory=list)

    def to_core(self):
        value = engine().ForwardModelMetadata()
        value.reference = to_core(self.reference)
        value.input_parameters = [to_core(p) for p in self.input_parameters]
        value.output_spectra = self.output_spectra
        value.preprocessing, value.postprocessing = self.preprocessing, self.postprocessing
        return value


@dataclass(slots=True)
class InverseModelMetadata:
    """Labelled measured spectra to a fixed layer/element depth profile.

    input_preparation contains per-measurement physical pipelines. preprocessing
    runs on the complete packed tensor. Uncertainty graphs pack means, presence
    probabilities, then standard deviations, each in layer-major order.
    """
    reference_setup: object
    input_spectra: object
    elements: list[str]
    layers: int
    uncertainty: bool = False
    input_preparation: list = field(default_factory=list)
    preprocessing: list = field(default_factory=list)
    postprocessing: list = field(default_factory=list)

    def to_core(self):
        value = engine().InverseModelMetadata()
        value.reference_setup = to_core(self.reference_setup)
        value.input_spectra = self.input_spectra
        value.elements, value.layers, value.uncertainty = self.elements, self.layers, self.uncertainty
        value.input_preparation = self.input_preparation
        value.preprocessing, value.postprocessing = self.preprocessing, self.postprocessing
        return value


@dataclass(slots=True)
class ModelMetadata:
    """Directional contract, graph tensor names and user provenance."""
    model: ForwardModelMetadata | InverseModelMetadata
    input_name: str = "inputs"
    output_name: str = "outputs"
    provenance: str = ""

    def to_core(self):
        value = engine().ModelMetadata(to_core(self.model))
        value.input_name, value.output_name, value.provenance = self.input_name, self.output_name, self.provenance
        return value

    def validate(self):
        self.to_core().validate()

    @property
    def direction(self):
        return self.to_core().direction

    def input_width(self):
        return self.to_core().input_width()

    def output_width(self):
        return self.to_core().output_width()

    @classmethod
    def from_core(cls, value):
        m = value.model
        core = engine()
        if isinstance(m, core.ForwardModelMetadata):
            directional = ForwardModelMetadata(experiment_from_core(m.reference),
                [ParameterSpec(p.name, p.target, p.unit) for p in m.input_parameters],
                m.output_spectra, m.preprocessing, m.postprocessing)
        else:
            directional = InverseModelMetadata(setup_from_core(m.reference_setup),
                m.input_spectra, m.elements, m.layers, m.uncertainty,
                m.input_preparation, m.preprocessing, m.postprocessing)
        return cls(directional, value.input_name, value.output_name, value.provenance)
