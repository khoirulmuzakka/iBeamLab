"""Smooth elemental profiles and reusable sublayer discretization.

Depths and layer widths use 1e15 atoms/cm2. Functions describe nonnegative
composition weights, normalized across species at each sublayer midpoint.
"""
from dataclasses import dataclass
from ._engine import engine, to_core
from .sample import Sample, sample_from_core


@dataclass(frozen=True, slots=True)
class ProfileFunction:
    """Function in normalized local depth y in [0,1].

    kind is constant, polynomial, or erf_polynomial. Polynomial coefficients are
    ascending powers. Erf coefficients are [A,a,b,c,p1,...], representing
    abs(A*(1+a*erf((y-b)/c))*(1+p1*y+...)); c must be positive.
    """
    kind: str
    coefficients: tuple

    def to_core(self):
        """Validate coefficients and copy into an owning native function."""
        n = engine()
        kinds = {"constant": n.ProfileFunctionKind.CONSTANT,
                 "polynomial": n.ProfileFunctionKind.POLYNOMIAL,
                 "erf_polynomial": n.ProfileFunctionKind.ERF_POLYNOMIAL}
        if self.kind not in kinds:
            raise ValueError(f"unknown profile function: {self.kind}")
        value = n.ProfileFunction()
        value.kind, value.coefficients = kinds[self.kind], list(self.coefficients)
        value.validate()
        return value

    def evaluate(self, depth, *, shift=0., width=1.):
        """Evaluate at physical depth; return zero outside [shift,shift+width]."""
        return self.to_core().evaluate(depth, shift, width)


def constant(value=1.):
    """Return a constant nonnegative composition-weight function."""
    return ProfileFunction("constant", (value,))


def polynomial(*coefficients):
    """Return a polynomial weight using ascending powers, followed by absolute value."""
    return ProfileFunction("polynomial", coefficients)


def erf_polynomial(amplitude, asymmetry, center, width, *coefficients):
    """Return an erf-polynomial weight; extra coefficients start with y**1."""
    return ProfileFunction("erf_polynomial", (amplitude, asymmetry, center, width, *coefficients))


@dataclass(frozen=True, slots=True)
class ProfileLayer:
    """Macro layer with one function per reference species, in species order.

    reference supplies thickness, isotope identities, roughness and porosity.
    sublayers is positive. spacing is linear or logarithmic (base-4 legacy grid).
    Reference fractions are an initial valid sample; functions determine the
    discretized fractions. Zero total function weight is rejected.
    """
    reference: object
    functions: tuple
    sublayers: int = 1
    spacing: str = "linear"

    def to_core(self):
        """Copy profile topology and reference layer into native storage."""
        n = engine()
        if self.spacing not in {"linear", "logarithmic"}:
            raise ValueError("spacing must be linear or logarithmic")
        value = n.ProfileLayer()
        value.reference = Sample([self.reference]).to_core().layers[0]
        value.functions = [to_core(f) for f in self.functions]
        value.sublayers = self.sublayers
        value.spacing = getattr(n.ProfileSpacing, self.spacing.upper())
        return value


@dataclass(frozen=True, slots=True)
class ProfileModel:
    """Ordered macro layers; discretizes to the ordinary Sample representation."""
    layers: tuple

    def to_core(self):
        """Validate and copy the physical profile description."""
        value = engine().ProfileModel()
        value.layers = [to_core(layer) for layer in self.layers]
        value.validate()
        return value

    def reference_sample(self):
        """Return the macro-layer reference sample before discretization."""
        return sample_from_core(self.to_core().reference_sample())

    def discretize(self):
        """Return an owning sublayer sample for simulation or inspection."""
        return sample_from_core(engine().discretize_profile(self.to_core()))


class Discretizer:
    """Compile grid edges once and retain output storage between evaluations.

    apply(model) returns an owning Python Sample. apply_into(model, native_sample)
    writes directly into caller-owned native storage without the Python copy.
    Changing coefficients or thickness is allowed; topology changes are rejected.
    """
    def __init__(self, model):
        self._core = engine().ProfileDiscretizer(to_core(model))
        self._output = engine().Sample()

    def apply_into(self, model, out):
        """Write to a native Sample; one discretizer instance belongs to one caller."""
        self._core.apply(to_core(model), out)

    def apply(self, model):
        """Evaluate into retained storage and return an independent Python Sample."""
        self.apply_into(model, self._output)
        return sample_from_core(self._output)
