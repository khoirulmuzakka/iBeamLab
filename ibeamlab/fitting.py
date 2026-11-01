"""Physical sample fitting with reusable native population objectives."""
from dataclasses import dataclass, field
import numpy as np
from ._engine import engine, to_core
from .exceptions import FitError
from .sample import experiment_from_core
from .simulator import result_from_core


@dataclass(frozen=True, slots=True)
class ArealDensity:
    """Bound one element's areal density in one zero-based layer.

    Units are 1e15 atoms/cm2. Total thickness is the sum of component densities;
    unspecified elements keep their initial density. Do not also vary thickness
    or fractions in the same layer. ``name`` defaults to ``ad.<layer>.<element>``.
    """
    layer: int
    element: str
    bounds: tuple
    name: str = ""


@dataclass(frozen=True, slots=True)
class Composition:
    """Vary N element fractions using N-1 bounded stick-breaking coordinates.

    ``bounds`` contains one physical (minimum, maximum) pair per element, or is
    empty for [0,1]. Elements outside the group keep their initial fractions.
    The group retains its initial total fraction. Result parameters report
    physical fractions; optimizer coordinates are unit-interval stick variables.
    """
    layer: int
    elements: tuple
    bounds: tuple = ()
    name_prefix: str = ""


@dataclass(frozen=True, slots=True)
class TiedParameter:
    """Apply one Parameter/ArealDensity coordinate to additional physical targets.

    Additional targets use core target descriptors or ArealDensity descriptions.
    All targets must have the same physical kind and initial value. The primary
    parameter supplies bounds and the result name.
    """
    parameter: object
    targets: tuple


@dataclass(frozen=True, slots=True)
class Region:
    """Select zero-based channels in [start, stop), with a nonnegative weight."""
    start: int
    stop: int
    weight: float = 1.


@dataclass(frozen=True, slots=True)
class SpectrumSelection:
    """Weight a measurement and optional disjoint channel regions.

    An empty regions sequence selects every valid observed channel. Measurement
    and region weights multiply. A zero weight excludes those channels.
    """
    measurement: str
    weight: float = 1.
    regions: tuple = ()


@dataclass(slots=True)
class FitProblem:
    """Describe an experimental fit independently of the optimizer.

    Parameters
    ----------
    data : MeasuredData
        Explicit measurement configurations and observations, without a sample.
    initial_sample : Sample
        Initial physical sample, including isotope assumptions.
    parameters : sequence
        Bounded ``vary.*`` parameters, ArealDensity, or TiedParameter objects.
        Equal bounds are fixed. Initial values must satisfy bounds.
    predictor : ForwardModel or Simulator
        Owned by the caller. A surrogate evaluates the whole valid population
        in one ONNX call. Use a separate model/objective per concurrent fit.
    compositions : sequence of Composition, optional
        Normalized fraction groups; no implicit fraction normalization occurs.
    selections : sequence of SpectrumSelection, optional
        Measurement weights and channel regions; unlisted measurements use all
        valid channels with weight one.
    loss : {"data_chi_square", "variance_chi_square"}
        Default is sum((data-prediction)**2 / (data_floor+data)). The variance
        loss requires explicitly supplied positive variances on selected channels.
    data_floor : float, default=1
        Positive denominator floor for the default loss.

    Notes
    -----
    Calibration, exposure, beam and geometry bounds use ``vary`` descriptors.
    The predictor's metadata must explicitly permit the requested changes.
    """
    data: object
    initial_sample: object
    parameters: object
    predictor: object
    compositions: tuple = ()
    selections: tuple = ()
    loss: str = "data_chi_square"
    data_floor: float = 1.

    def to_core(self):
        """Validate and compile the declarative inputs to native value types."""
        n = engine()
        value = n.FitProblem()
        value.data, value.initial_sample = to_core(self.data), to_core(self.initial_sample)
        def target(p):
            return n.ElementArealDensity(p.layer, p.element) if isinstance(p, ArealDensity) else p
        parameters = []
        for description in self.parameters:
            primary = description.parameter if isinstance(description, TiedParameter) else description
            p = n.FitParameter()
            if isinstance(primary, ArealDensity):
                p.name = primary.name or f"ad.{primary.layer}.{primary.element}"
                p.target = target(primary)
            else:
                p.name, p.target = primary.name, primary.target
            p.bounds = n.ParameterBounds(*primary.bounds)
            if isinstance(description, TiedParameter):
                p.tied_targets = [target(t) for t in description.targets]
            parameters.append(p)
        value.parameters = parameters
        compositions = []
        for c in self.compositions:
            native = n.FitComposition()
            native.layer, native.elements, native.name_prefix = c.layer, list(c.elements), c.name_prefix
            native.bounds = [n.ParameterBounds(*b) for b in c.bounds]
            compositions.append(native)
        value.compositions = compositions
        selections = []
        for s in self.selections:
            native = n.FitSelection()
            native.measurement, native.weight = s.measurement, s.weight
            native.regions = [n.FitRegion(r.start, r.stop, r.weight) for r in s.regions]
            selections.append(native)
        value.selections = selections
        losses = {"data_chi_square": n.FitLoss.DATA_CHI_SQUARE, "variance_chi_square": n.FitLoss.VARIANCE_CHI_SQUARE}
        if self.loss not in losses:
            raise FitError(f"unknown loss: {self.loss}")
        value.loss, value.data_floor = losses[self.loss], self.data_floor
        return value


class FitResult:
    """Best evaluated physical solution, prediction, chi-square and diagnostics.

    ``sample`` and ``setup`` are the fitted physical descriptions. ``parameters``
    maps physical names to values, while ``coordinates`` contains solver values.
    ``contributions`` reports weighted summed cost and channel count per detector.
    ``prediction`` is the already-evaluated best spectrum; cancellation never
    recomputes it. ``history`` contains scalar progress records only.
    """
    def __init__(self, native):
        self._core = native
        self.experiment = experiment_from_core(native.experiment)
        self.sample, self.setup = self.experiment.sample, self.experiment.setup
        self.prediction = result_from_core(native.prediction)
        self.parameters = dict(zip(native.parameter_names, native.parameter_values))
        self.coordinates = np.asarray(native.coordinates)
        for name in ("cost", "evaluations", "invalid", "status", "message", "contributions", "history", "feasible"):
            setattr(self, name, getattr(native, name))


class FitObjective:
    """Compile a FitProblem once and retain buffers across changing populations.

    Construction evaluates the initial solution once and includes it in the
    evaluation count. The object retains its predictor and input descriptions.
    Methods are not concurrent; only request_stop is safe from another thread.
    """
    def __init__(self, problem):
        from .models import ForwardModel
        from .simulator import Simulator
        self.problem = problem
        n = engine()
        try:
            if isinstance(problem.predictor, ForwardModel):
                self._predictor = n.ForwardFitPredictor(problem.predictor._core)
            elif isinstance(problem.predictor, Simulator):
                self._backend = problem.predictor.to_core()
                self._predictor = n.SimulatorFitPredictor(self._backend)
            else:
                raise TypeError("predictor must be a ForwardModel or Simulator")
            self._core = n.FitObjective(problem.to_core(), self._predictor)
        except (ValueError, RuntimeError) as error:
            raise FitError(str(error)) from error

    @property
    def dimension(self):
        """Number of free optimizer coordinates, excluding fixed parameters."""
        return self._core.dimension

    @property
    def bounds(self):
        """Inclusive bounds in optimizer coordinates, shaped [dimension,2]."""
        return np.asarray(self._core.bounds, dtype=np.float64).reshape(-1, 2)

    @property
    def coordinate_names(self):
        """Ordered solver column names; composition columns are latent sticks."""
        return tuple(self._core.coordinate_names)

    @property
    def initial_coordinates(self):
        """Initial optimizer coordinate vector as an independent float64 array."""
        return np.asarray(self._core.initial_coordinates, dtype=np.float64)

    def evaluate_batch(self, candidates, *, out=None):
        """Score [population,dimension] rows, optionally into reusable costs.

        Inputs are converted to contiguous float64 when needed. ``out`` must
        already be writable contiguous float64 with shape [population]. Invalid
        candidates receive infinity; configuration/backend failures raise.
        Valid surrogate rows are predicted together, including reduced populations.
        """
        rows = np.ascontiguousarray(candidates, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != self.dimension:
            raise ValueError("candidates must have shape [population, dimension]")
        if out is None:
            out = np.empty(rows.shape[0], dtype=np.float64)
        if not isinstance(out, np.ndarray) or out.dtype != np.float64 or out.shape != (rows.shape[0],) or not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError("out must be writable contiguous float64 [population]")
        try:
            self._core.evaluate_batch(rows, out)
        except (ValueError, RuntimeError) as error:
            raise FitError(str(error)) from error
        return out

    def decode(self, coordinates):
        """Return the physical experiment represented by one feasible row."""
        return experiment_from_core(self._core.decode(coordinates))

    def best_result(self):
        """Copy the best solution evaluated so far, including its prediction."""
        return FitResult(self._core.best_result())

    def progress(self):
        """Return evaluations, invalid count and best cost as scalar statistics."""
        return self._core.progress()

    @property
    def stopped(self):
        """Whether cooperative cancellation has been requested."""
        return self._core.stopped

    def request_stop(self):
        """Request cancellation; the best completed prediction is preserved."""
        self._core.request_stop()


def fit(problem, *, optimizer="arrde", max_evaluations=10000,
        population_size=0, seed=42, record_history=True, progress=None):
    """Fit an unknown sample with population-based physical prediction.

    Parameters
    ----------
    problem : FitProblem or FitObjective
        Declarative problem or an existing compiled objective.
    optimizer : "arrde" or callable
        Native ARRDE, or ``optimizer(objective, options)``. A custom optimizer
        calls evaluate_batch and returns None; the best evaluated solution wins.
    max_evaluations : int, default=10000
        Total candidate budget including the initial preflight prediction.
    population_size : int, default=0
        ARRDE population size; zero uses the upstream automatic size.
    seed : int, default=42
        ARRDE random seed.
    record_history : bool, default=True
        Retain scalar progress history, never candidate populations.
    progress : callable, optional
        Called with native progress statistics. Return True to stop gracefully.

    Returns
    -------
    FitResult
        Best already-evaluated solution. Keyboard interruption preserves it.

    Notes
    -----
    Use ``ibeamlab.training.fit`` for neural-network training. Native ARRDE can
    be disabled at build time; external optimizers remain supported.
    """
    objective = problem if isinstance(problem, FitObjective) else FitObjective(problem)
    options = dict(max_evaluations=max_evaluations, population_size=population_size,
                   seed=seed, record_history=record_history, progress=progress)
    try:
        if callable(optimizer):
            optimizer(objective, options)
            return objective.best_result()
        if optimizer != "arrde":
            raise ValueError(f"unknown optimizer: {optimizer}")
        settings = engine().FitOptions()
        for name, value in options.items():
            if name != "progress":
                setattr(settings, name, value)
        return FitResult(engine().fit_arrde(objective._core, settings, progress))
    except KeyboardInterrupt:
        objective.request_stop()
        return objective.best_result()
    except (ValueError, RuntimeError) as error:
        raise FitError(str(error)) from error
