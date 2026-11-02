"""Measurement-specific prediction and concurrent independent fits."""
from dataclasses import dataclass, field
from ._engine import engine
from .fitting import FitResult


def as_predictor(provider):
    """Adapt an owning Python model/simulator or composite provider to fitting."""
    from .models import ForwardModel
    from .simulator import Simulator
    n = engine()
    if isinstance(provider, ForwardModel):
        return n.ForwardFitPredictor(provider._core)
    if isinstance(provider, Simulator):
        return n.SimulatorFitPredictor(provider.to_core())
    if isinstance(provider, CompositePredictor):
        return provider._core
    if isinstance(provider, n.FitPredictor):
        return provider
    raise TypeError("predictor must be a forward model, simulator, or CompositePredictor")


class CompositePredictor:
    """Route measurement groups to distinct providers sharing one physical sample.

    routes maps tuples of labels to ForwardModel/Simulator instances. Every input
    measurement must be routed exactly once. Each provider sees only its group's
    setup and evaluates the population in one batch. Retains provider ownership;
    use separate composite/provider instances for concurrent fits.
    """
    def __init__(self, routes):
        n = engine()
        self.providers = tuple(routes.values())
        self._adapters = []
        values = []
        for labels, provider in routes.items():
            route = n.AnalysisRoute()
            route.measurements = [labels] if isinstance(labels, str) else list(labels)
            adapter = as_predictor(provider)
            self._adapters.append(adapter)
            route.predictor = adapter
            values.append(route)
        plan = n.AnalysisPlan()
        plan.routes = values
        plan.validate()
        self._core = n.AnalysisPredictor(plan)


@dataclass(frozen=True, slots=True)
class AnalysisTask:
    """Named independent fit; initial_from names earlier tasks supplying guesses.

    problem contains scientific settings. predictor assignment comes from the
    executor worker factory. options accepts the same native optimizer settings
    as fit(), excluding Python callbacks and external callables.
    """
    id: str
    problem: object
    options: dict = field(default_factory=dict)
    initial_from: tuple = ()


@dataclass(frozen=True, slots=True)
class AnalysisOutcome:
    """Ordered task result and diagnostic; a nonempty error indicates failure."""
    id: str
    result: object
    error: str
    status: str = ""


class AnalysisExecutor:
    """Retain separate predictor/session resources across many independent fits.

    predictor_factory(worker_index) creates a distinct provider for each worker;
    choose CPU/GPU/thread settings there. Native fitting releases the GIL.
    Progress callbacks receive (task_id, worker_index, statistics) serially;
    returning True stops that task. request_stop() cancels all active/pending tasks.
    Recreate the executor after cancellation. Concurrent run() calls are unsupported.
    """
    def __init__(self, workers, predictor_factory):
        if type(workers) is not int or not 1 <= workers <= 256:
            raise ValueError("workers must be an integer in [1,256]")
        self.providers = [predictor_factory(i) for i in range(workers)]
        if len({id(p) for p in self.providers}) != workers:
            raise ValueError("each worker requires a distinct provider")
        self._adapters = [as_predictor(p) for p in self.providers]
        self._core = engine().AnalysisExecutor(self._adapters)

    def run(self, tasks, *, progress=None):
        """Fit tasks and return outcomes in input order; preserve task-local errors."""
        from .fitting import _optimizer_options
        n = engine()
        descriptions = tuple(tasks)  # Retain problem/backend ownership during execution.
        native = []
        for task in descriptions:
            value = n.AnalysisTask()
            value.id, value.problem = task.id, task.problem.to_core()
            options = dict(task.options)
            stages = options.pop("stages", ())
            value.options = _optimizer_options(**options)
            value.stages = [_optimizer_options(**stage) for stage in stages]
            value.initial_from = list(task.initial_from)
            native.append(value)
        return [AnalysisOutcome(v.id, FitResult(v.result) if not v.error and v.result.feasible else None, v.error, v.result.status)
                for v in self._core.run(native, progress)]

    def request_stop(self):
        """Request cooperative cancellation without discarding completed outcomes."""
        self._core.request_stop()
