# Measurement routing and parallel analyses

An analysis can share one physical sample across measurements using different
simulators or surrogate packages. `CompositePredictor` routes each measurement
group to a provider and retains population batching:

```python
predictor = ibl.CompositePredictor({
    ("RBS",): rbs_forward_model,
    ("ERDA",): erda_simulator,
})
problem = ibl.FitProblem(data, initial_sample, parameters, predictor)
result = ibl.fit(problem)
```

Every measurement is routed exactly once. A provider receives the shared sample
and only its route's measurement setup. A route can include multiple detector
labels belonging to one model/backend. Outputs are restored to the input setup's
measurement order. A varying shared-sample parameter must be accepted by every
provider; a measurement-specific parameter is checked against its own provider.

C++ uses `analysis::Plan`, `analysis::Route`, and `analysis::Predictor`. Each route
retains independent scratch arrays at its peak population capacity. Smaller
populations reuse them, and different detector layouts do not discard each
other's spectrum buffers. Sample/species topology changes can still allocate.
Calls on one predictor must be serialized. Predictor failures remain attached to
the corresponding candidate; execution errors raise diagnostics.

See [the multi-measurement example](../examples/Fitting/fit_multiple_measurements.py).

## Independent fits

```python
executor = ibl.AnalysisExecutor(
    workers=2,
    predictor_factory=lambda worker: ibl.load_model("forward.ibm"),
)
tasks = [
    ibl.AnalysisTask("sample-a", problem_a, {"max_evaluations": 1000}),
    ibl.AnalysisTask("sample-b", problem_b, {"max_evaluations": 1000},
                     initial_from=("sample-a",)),
]
outcomes = executor.run(tasks)
```

The executor owns distinct worker adapters and retains the Python providers.
The factory selects CPU/CUDA devices and internal thread budgets through the
ordinary inference/backend settings. Each worker retains its predictor resources
between tasks; each task compiles its own data and objective. The task problem's
predictor is ignored in favor of the assigned worker provider.

Results preserve input order and IDs. Errors are recorded per task. Initialization
dependencies must refer to earlier tasks, preventing cycles; completed feasible
results seed a dependent fit only when coordinate names match and values satisfy
the new bounds. Task options also accept a `stages` list for staged execution. Tasks with a
failed dependency still run using their own initial settings.

`progress(task_id, worker_index, statistics)` callbacks are serialized. Return
true to stop that fit. `request_stop()` cooperatively stops all active and queued
tasks while preserving completed results. Recreate an executor after global
cancellation. Do not call `run()` concurrently on the same executor. Avoid sharing
an underlying model/session between worker factories, including inside composites.

C++ exposes the same behavior through `analysis::Executor`, with a predictor
factory taking a worker index. No GUI or application project types are involved.

## Optimizer stages

Built-in optimizers are ARRDE and DE. Both accept settings dictionaries and
multiple bounded `initial_guesses`. Unsupported native algorithm names fail
explicitly; external optimizers can use `FitObjective.evaluate_batch`.

```python
result = ibl.fit(problem, stages=[
    dict(optimizer="de", max_evaluations=2000, population_size=64, seed=42),
    dict(optimizer="arrde", max_evaluations=5000, seed=43),
])
```

Stages run on one objective. The first budget includes its initial prediction;
later budgets are additional. Best evaluated coordinates seed subsequent stages.
Each stage has its own settings, seed and history control. Cancellation retains
one matching best sample, physical parameter vector, and prediction.

## Instrument import and scientific output

`SpectrumParser` adds start/end data markers and header acquisition-time fields
to the numeric text/CSV reader. `TimeField` supplies a marker, seconds-per-unit
conversion, and optional fallback in seconds. Missing channels stay masked unless
explicitly declared zero. Parsing does not infer beam settings or geometry.

`write_fit_result(path, result, provenance=...)` writes a new directory containing
sample/isotope, setup, parameter, optimizer-coordinate, spectrum, elemental-spectrum,
objective, history and provenance tables. It refuses to overwrite an existing
directory. Application project persistence remains outside this API.

SIMNRA optionally captures elemental spectra using `capture_elemental=True`.
Inspect `result.elemental_spectra[measurement][element]`. These diagnostics can
be exported with fit results; ordinary dataset shards store total spectra only
and reject records carrying elemental diagnostics. `export_project()`
recalculates a single-measurement experiment on a dedicated COM thread and saves
a new project, optionally inserting measured counts. This is an on-demand export,
not part of population evaluation. `capabilities` reports backend support.
