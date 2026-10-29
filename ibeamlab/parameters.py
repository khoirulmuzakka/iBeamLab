"""Ordered physical parameter descriptions shared by generation and model metadata."""
from dataclasses import dataclass
from ._engine import engine
from .exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """One named physical input column, with a C++ parameter target and unit label."""
    name: str
    target: object
    unit: str = ""

    def to_core(self):
        if not self.name:
            raise ValidationError("parameter name cannot be empty")
        return engine().Parameter(self.name, self.target, self.unit)


@dataclass(frozen=True, slots=True)
class Parameter(ParameterSpec):
    """Physical input column with inclusive finite bounds for independent sampling."""
    bounds: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self):
        import math
        values = tuple(self.bounds)
        if len(values) != 2 or not all(math.isfinite(v) for v in values) or values[0] > values[1]:
            raise ValidationError("parameter bounds must be finite and ordered")
        object.__setattr__(self, "bounds", values)


class _Vary:
    def layer_thickness(self, *, layer, bounds, name=None):
        return Parameter(name or f"layer_{layer}_thickness", engine().LayerThickness(layer), "1e15 atoms/cm2", bounds)

    def concentration(self, element, *, layer, bounds, name=None):
        return Parameter(name or f"layer_{layer}_{element}", engine().SpeciesConcentration(layer, element), "fraction", bounds)

    def _measurement(self, kind, measurement, bounds, name, unit):
        return Parameter(name or f"{measurement}_{kind}", getattr(engine(), kind)(measurement), unit, bounds)

    def beam_energy(self, measurement, *, bounds, name=None):
        return self._measurement("BeamEnergy", measurement, bounds, name, "keV")

    def beam_spread(self, measurement, *, bounds, name=None):
        return self._measurement("BeamSpread", measurement, bounds, name, "keV")

    def detector_resolution(self, measurement, *, bounds, name=None):
        return self._measurement("DetectorResolution", measurement, bounds, name, "keV")

    def particles_sr(self, measurement, *, bounds, name=None):
        return self._measurement("ParticlesSr", measurement, bounds, name, "particles*sr")

    def calibration_linear(self, measurement, *, bounds, name=None):
        return self._measurement("CalibrationLinear", measurement, bounds, name, "keV/channel")

    def calibration_offset(self, measurement, *, bounds, name=None):
        return self._measurement("CalibrationOffset", measurement, bounds, name, "keV")

    def calibration_quadratic(self, measurement, *, bounds, name=None):
        return self._measurement("CalibrationQuadratic", measurement, bounds, name, "keV/channel2")


vary = _Vary()
