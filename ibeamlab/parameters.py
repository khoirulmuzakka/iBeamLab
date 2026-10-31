"""Ordered physical parameter descriptions shared by generation and model metadata."""
from __future__ import annotations
from dataclasses import dataclass
from typing import TypeAlias, Union
from . import native as core
from ._engine import engine
from .exceptions import ValidationError

ParameterTarget: TypeAlias = Union["core.LayerThickness", "core.SpeciesConcentration", "core.BeamEnergy",
    "core.BeamSpread", "core.CalibrationLinear", "core.CalibrationOffset", "core.CalibrationQuadratic",
    "core.DetectorResolution", "core.ParticlesSr", "core.IncidenceAngle", "core.ExitAngle", "core.ScatteringAngle"]


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Declare one ordered physical parameter column.

    Parameters
    ----------
    name : str
        Nonempty unique column name.
    target : ParameterTarget
        Physical location, such as LayerThickness(0) or ScatteringAngle("RBS").
    unit : str, default=""
        Descriptive unit label. Values must use the physical units of the target;
        this string does not perform unit conversion.

    Notes
    -----
    Parameter names and target identities must be unique within a model or plan.
    Full validation happens when converting the enclosing plan or metadata.
    """
    name: str
    target: ParameterTarget
    unit: str = ""

    def to_core(self) -> core.Parameter:
        """Convert this description to an owning native value.

        Returns
        -------
        ibeamlab.native.Parameter
            Native value for lower-level workflows.

        Notes
        -----
        Creates a native parameter descriptor and preserves conversion errors as
        ValidationError with field context.
        """
        if not isinstance(self.name, str) or not self.name:
            raise ValidationError("ParameterSpec.name must be a nonempty string")
        if not isinstance(self.unit, str):
            raise ValidationError("ParameterSpec.unit must be a string")
        try:
            return engine().Parameter(self.name, self.target, self.unit)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValidationError(f"ParameterSpec.target: {error}") from error


@dataclass(frozen=True, slots=True)
class Parameter(ParameterSpec):
    """Declare a physical parameter with inclusive sampling bounds.

    Parameters
    ----------
    name : str
        Column name; see ParameterSpec.
    target : ParameterTarget
        Physical location; see ParameterSpec.
    unit : str, default=""
        Descriptive unit label.
    bounds : tuple[float, float], default=(0.0, 1.0)
        Finite inclusive lower and upper limits; lower must not exceed upper.

    Raises
    ------
    ValidationError
        Bounds are invalid.

    Notes
    -----
    Bounds drive GenerationPlan.sample_uniform(). Generation from supplied rows
    validates physical values but does not enforce these Python sampling bounds.
    """
    bounds: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self):
        import math
        values = tuple(self.bounds)
        if len(values) != 2 or not all(math.isfinite(v) for v in values) or values[0] > values[1]:
            raise ValidationError("parameter bounds must be finite and ordered")
        object.__setattr__(self, "bounds", values)


class _Vary:
    """Create bounded physical parameter columns through the public vary object.

    Notes
    -----
    Layer indices are zero-based. Measurement parameters select labels, not
    measurement positions. Fractions and coupled geometry require coordinated rows;
    independent uniform sampling does not enforce those cross-parameter constraints.
    """
    def layer_thickness(self, *, layer, bounds, name=None):
        """Create a bounded column for layer areal thickness.

        Parameters
        ----------
        layer : int
            Zero-based layer index.
        bounds : tuple[float, float]
            Inclusive finite lower and upper sampling limits in 1e15 atoms/cm2.
        name : str, optional
            Explicit column name; a target-based name is generated when omitted.

        Returns
        -------
        Parameter
            Descriptor suitable for GenerationPlan.parameters.
        """
        return Parameter(name or f"layer_{layer}_thickness", engine().LayerThickness(layer), "1e15 atoms/cm2", bounds)

    def concentration(self, element, *, layer, bounds, name=None):
        """Create a bounded column for element atomic fraction.

        Parameters
        ----------
        element : str
            Canonical element symbol.
        layer : int
            Zero-based layer index.
        bounds : tuple[float, float]
            Inclusive finite lower and upper sampling limits in fraction.
        name : str, optional
            Explicit column name; a target-based name is generated when omitted.

        Returns
        -------
        Parameter
            Descriptor suitable for GenerationPlan.parameters.

        Notes
        -----
        Concentration rows must jointly sum to one per layer.
        """
        return Parameter(name or f"layer_{layer}_{element}", engine().SpeciesConcentration(layer, element), "fraction", bounds)

    def _measurement(self, kind, measurement, bounds, name, unit):
        return Parameter(name or f"{measurement}_{kind}", getattr(engine(), kind)(measurement), unit, bounds)

    def beam_energy(self, measurement, *, bounds, name=None):
        """Create a bounded column for incident beam energy.

        Parameters
        ----------
        measurement : str
            Measurement label.
        bounds : tuple[float, float]
            Inclusive finite lower and upper sampling limits in keV.
        name : str, optional
            Explicit column name; a target-based name is generated when omitted.

        Returns
        -------
        Parameter
            Descriptor suitable for GenerationPlan.parameters.
        """
        return self._measurement("BeamEnergy", measurement, bounds, name, "keV")

    def beam_spread(self, measurement, *, bounds, name=None):
        """Create a bounded column for beam energy spread.

        Parameters
        ----------
        measurement : str
            Measurement label.
        bounds : tuple[float, float]
            Inclusive finite lower and upper sampling limits in keV FWHM.
        name : str, optional
            Explicit column name; a target-based name is generated when omitted.

        Returns
        -------
        Parameter
            Descriptor suitable for GenerationPlan.parameters.
        """
        return self._measurement("BeamSpread", measurement, bounds, name, "keV")

    def detector_resolution(self, measurement, *, bounds, name=None):
        """Create a bounded column for detector energy resolution.

        Parameters
        ----------
        measurement : str
            Measurement label.
        bounds : tuple[float, float]
            Inclusive finite lower and upper sampling limits in keV FWHM.
        name : str, optional
            Explicit column name; a target-based name is generated when omitted.

        Returns
        -------
        Parameter
            Descriptor suitable for GenerationPlan.parameters.
        """
        return self._measurement("DetectorResolution", measurement, bounds, name, "keV")

    def particles_sr(self, measurement, *, bounds, name=None):
        """Create a bounded column for particle-count times solid angle.

        Parameters
        ----------
        measurement : str
            Measurement label.
        bounds : tuple[float, float]
            Inclusive finite lower and upper sampling limits in particles*sr.
        name : str, optional
            Explicit column name; a target-based name is generated when omitted.

        Returns
        -------
        Parameter
            Descriptor suitable for GenerationPlan.parameters.
        """
        return self._measurement("ParticlesSr", measurement, bounds, name, "particles*sr")

    def calibration_linear(self, measurement, *, bounds, name=None):
        """Create a bounded column for linear channel calibration.

        Parameters
        ----------
        measurement : str
            Measurement label.
        bounds : tuple[float, float]
            Inclusive finite lower and upper sampling limits in keV/channel.
        name : str, optional
            Explicit column name; a target-based name is generated when omitted.

        Returns
        -------
        Parameter
            Descriptor suitable for GenerationPlan.parameters.
        """
        return self._measurement("CalibrationLinear", measurement, bounds, name, "keV/channel")

    def calibration_offset(self, measurement, *, bounds, name=None):
        """Create a bounded column for channel calibration offset.

        Parameters
        ----------
        measurement : str
            Measurement label.
        bounds : tuple[float, float]
            Inclusive finite lower and upper sampling limits in keV.
        name : str, optional
            Explicit column name; a target-based name is generated when omitted.

        Returns
        -------
        Parameter
            Descriptor suitable for GenerationPlan.parameters.
        """
        return self._measurement("CalibrationOffset", measurement, bounds, name, "keV")

    def calibration_quadratic(self, measurement, *, bounds, name=None):
        """Create a bounded column for quadratic channel calibration.

        Parameters
        ----------
        measurement : str
            Measurement label.
        bounds : tuple[float, float]
            Inclusive finite lower and upper sampling limits in keV/channel2.
        name : str, optional
            Explicit column name; a target-based name is generated when omitted.

        Returns
        -------
        Parameter
            Descriptor suitable for GenerationPlan.parameters.
        """
        return self._measurement("CalibrationQuadratic", measurement, bounds, name, "keV/channel2")

    def incidence_angle(self, measurement, *, bounds, name=None):
        """Create a bounded column for detector incidence angle alpha.

        Parameters
        ----------
        measurement : str
            Measurement label.
        bounds : tuple[float, float]
            Inclusive finite lower and upper sampling limits in degree.
        name : str, optional
            Explicit column name; a target-based name is generated when omitted.

        Returns
        -------
        Parameter
            Descriptor suitable for GenerationPlan.parameters.

        Notes
        -----
        Angle columns must jointly describe physically consistent geometry.
        """
        return self._measurement("IncidenceAngle", measurement, bounds, name, "degree")

    def exit_angle(self, measurement, *, bounds, name=None):
        """Create a bounded column for detector exit angle beta.

        Parameters
        ----------
        measurement : str
            Measurement label.
        bounds : tuple[float, float]
            Inclusive finite lower and upper sampling limits in degree.
        name : str, optional
            Explicit column name; a target-based name is generated when omitted.

        Returns
        -------
        Parameter
            Descriptor suitable for GenerationPlan.parameters.

        Notes
        -----
        Angle columns must jointly describe physically consistent geometry.
        """
        return self._measurement("ExitAngle", measurement, bounds, name, "degree")

    def scattering_angle(self, measurement, *, bounds, name=None):
        """Create a bounded column for detector scattering angle theta.

        Parameters
        ----------
        measurement : str
            Measurement label.
        bounds : tuple[float, float]
            Inclusive finite lower and upper sampling limits in degree.
        name : str, optional
            Explicit column name; a target-based name is generated when omitted.

        Returns
        -------
        Parameter
            Descriptor suitable for GenerationPlan.parameters.

        Notes
        -----
        Angle columns must jointly describe physically consistent geometry.
        """
        return self._measurement("ScatteringAngle", measurement, bounds, name, "degree")


vary = _Vary()
