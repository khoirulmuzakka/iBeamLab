# Fitting measured spectra

Experimental data belongs to `MeasuredData`: an explicit measurement setup and
one labelled `MeasuredSpectrum` per measurement. The unknown sample belongs to
`FitProblem`. Measurements may use different beam energies, detector geometries,
calibrations and acquisition conditions while sharing the fitted sample.

```python
import ibeamlab as ibl

observed = ibl.read_spectrum("rbs.txt", measurement="RBS",
                             format="channel_counts", channel_base=0)
data = ibl.MeasuredData(setup=setup, spectra=[observed])
problem = ibl.FitProblem(
    data=data,
    initial_sample=sample,
    parameters=[ibl.vary.layer_thickness(layer=0, bounds=(50, 500))],
    predictor=surrogate,
)
result = ibl.fit(problem, optimizer="arrde", max_evaluations=10000)
print(result.parameters, result.cost, result.status)
```

`setup`, `sample` and `surrogate` above are the ordinary framework descriptions
and `ForwardModel` instance. A `Simulator`, including a Python-defined backend,
can replace the surrogate. See [the runnable CLI example](../examples/Fitting/fit_spectrum.py).
Neural-network training uses `ibeamlab.training.fit`.

## Reading observations

Built-in readers support `counts` (one column), `channel_counts` (whitespace
channel/count columns), and `csv`. Column indices are always zero-based. Use
`header`, `skip_rows`, `delimiter`, `comment`, `counts_column`, `channel_column`,
`variance_column` and `channel_base` explicitly. Setting `channel_column=None`
reads sequential rows; setting `channel_base=1` converts one-based channels.
CSV fields may be quoted, with doubled quotes; multiline fields are unsupported.
Parse errors report the file and physical line. Duplicate, negative and fractional
channels are rejected. Missing channels retain their positions and have a false
validity mask; observed zeros remain valid. Readers never infer calibration or
geometry. `register_spectrum_reader` adds Python readers that return the same
`MeasuredSpectrum` contract.

## Parameters and constraints

Use bounded `vary` parameters for thickness, physical fractions, beam energy,
spread, calibration, exposure, resolution and detector angles. Equal bounds fix
a parameter and remove its optimizer coordinate. Initial values must satisfy the
bounds; they are never silently clipped. Geometry must remain physically
consistent for every candidate. Surrogate metadata must permit each varied field;
physical corrections cannot change an undeclared beam energy or detector angle.

Use `Composition(layer=0, elements=["Si", "O"], bounds=[(.2,.7),(.3,.8)])` in
`FitProblem.compositions` to vary a normalized composition. The group uses N-1
unit-interval stick-breaking coordinates and preserves its initial total fraction.
Elements outside the group retain their initial fractions. Results contain physical
fractions separately from solver coordinates. Independent fraction parameters do
not trigger normalization; physically invalid candidates receive infinite cost.

Use `ArealDensity(0, "Si", (0, 300))` in `parameters` to vary component areal
densities in 1e15 atoms/cm². Layer thickness is derived as their sum, and fractions
as density/total. Unspecified components retain their initial areal density. A
zero-density layer keeps its initial fractions. Do not also vary its thickness or
fractions. `TiedParameter(primary, targets)` applies one coordinate to several
physical targets of the same kind and initial value.

## Objective and batching

The default cost is the summed chi-square
`sum((data - prediction)**2 / (1 + data))`. `data_floor` changes the positive
floor. Known variances are used only when `loss="variance_chi_square"` is explicitly
selected. `SpectrumSelection` supplies detector weights and disjoint `Region`
intervals `[start, stop)`. Masks, regions and denominator coefficients are compiled
once. Extra predicted channels are ignored, while every selected observed channel
must be covered by the prediction. Detector contributions report both summed
weighted cost and selected channel count.

`FitObjective(problem)` evaluates the initial solution once, then retains input,
output and scoring buffers as the population shrinks or grows. Its
`evaluate_batch(rows, out=costs)` interface accepts contiguous float64
`[population, dimension]` coordinates and can fill caller-owned costs. Invalid
physical candidates are compacted before prediction and scattered back as infinity.
A forward model evaluates the valid population in one batch. Scoring does not
allocate a population-sized residual array. Buffers grow when a previous capacity
is exceeded; ONNX Runtime and simulator internals have their own allocation needs.

ARRDE is the pinned upstream Minion implementation, enabled by
`IBEAMLAB_ENABLE_ARRDE=ON`. Only its required native sources are embedded, without
an additional DLL or installed CMake dependency. Disable it to use the objective
with an external optimizer. A custom Python optimizer is callable as
`optimizer(objective, options)`; it evaluates candidates and returns None, and
`fit` returns the best solution the objective actually evaluated.

Progress callbacks receive scalar counts and best cost; returning True requests
cancellation. Keyboard interruption also preserves the best completed solution,
including its matching prediction. The initial evaluation counts toward the
budget. Histories store scalar statistics, not spectra or populations. A separate
objective and model session are required for each concurrent fit. CPU/GPU provider
and session thread settings use the existing `InferenceOptions`; fitting creates
no hidden inference thread pool.

The C++ equivalents are in `core/data.h`, `io/reader.h` and `fitting/fit.h`.
`fitting::Objective` owns the compiled problem and references a shared batch
predictor. `ForwardPredictor` and `SimulatorPredictor` borrow backend instances,
which must outlive the objective; Python retains their owners automatically.
