# Scientific backend TODO

Goal: make iBeamLab a self-sufficient scientific backend for AutoNRA, usable
independently through C++ and Python. AutoNRA provides the GUI and translates
projects into standalone iBeamLab settings.

## Design boundaries

- Keep the ordered stack of homogeneous layers as the simulation representation.
- Support smooth profiles as an additional representation that discretizes into
  that stack. Explicit stacks remain usable without a profile model.
- Keep Qt, AutoNRA project/configuration classes, and GUI state outside iBeamLab.
- Preserve AutoNRA's established numerical behavior when moving functionality.
  Make differing scientific policies explicit rather than silently changing them.
- Resolve bindings once and reuse worker-owned buffers during optimization.
- Expose new scientific capabilities in both C++ and Python, with documented units,
  shapes, ownership, and failure behavior.

## Implementation status

The profile, fitting-extension, measurement-routing, executor, parser and export
APIs are implemented in iBeamLab. See [profiles.md](profiles.md),
[analysis.md](analysis.md), and [fitting.md](fitting.md). Native and Python tests
cover analytic reference cases, profile fitting, population reduction, linked
parameters, ROI policies, reader metadata and worker cancellation.

AutoNRA integration is intentionally out of scope for this implementation.
End-to-end parity against AutoNRA project files remains to be established. Native
optimizer stages currently support ARRDE and DE; other optimizers can use the
external objective API. Instrument parser descriptors are standalone iBeamLab
settings, not AutoNRA parser/project files.

## 1. Establish reference cases

- [ ] Inventory active AutoNRA backend features and their existing tests.
- [ ] Select small representative cases: explicit layers, smooth profiles,
  multiple measurements, isotope mixtures, linked parameters, and SIMNRA export.
- [ ] Record expected profile values, discretized stacks, predictions, objective
  values, and parameter mappings from the current AutoNRA implementation.
- [ ] Separate numerical equivalence checks from optimizer convergence checks;
  record seeds, settings, and tolerances where optimization is compared.
- [ ] Keep SIMNRA-dependent checks optional on runners without SIMNRA.

Completion: reference cases identify which behaviors must remain equivalent and
can be evaluated without running production-scale fits.

## 2. Add profile parameterization and discretization

Suggested home: `profile`.

- [x] Port constant, polynomial, and error-function/polynomial profile functions
  from AutoNRA's `Parameterization` implementation.
- [x] Preserve depth shifts, widths, coefficient meanings, and units.
- [x] Port sublayer counts and existing discretization schemes from `LayerModel`.
- [x] Preserve profile evaluation locations, normalization, and layer-boundary
  behavior, including roughness and porosity handling.
- [x] Validate invalid profiles and discretization settings explicitly.
- [x] Evaluate into caller-owned sample storage; retain sublayer/species capacity.
- [x] Provide C++ and Python APIs for profile inspection and discretization.

Completion: the same profile parameters produce equivalent sublayer stacks in
AutoNRA and iBeamLab, without requiring AutoNRA configuration types.

## 3. Generalize fitting parameter mapping

Suggested home: `fitting`, using `profile` and existing physical targets.

- [x] Add mappings from optimizer coordinates to profile coefficients.
- [x] Support fixed values, bounds, equal-value ties, and ratio-linked parameters.
- [x] Preserve bounded, normalized composition handling.
- [x] Expose roughness and porosity as supported physical fit targets.
- [x] Preserve shared sample parameters across independently configured measurements.
- [x] Validate incompatible links, duplicate assignments, and infeasible bounds
  before optimization.
- [x] Compile bindings once; avoid parameter-name maps and repeated string lookups
  in the population evaluation path.
- [x] Return named physical parameters alongside optimizer coordinates.

Completion: AutoNRA's supported parameterizations and constraints can be expressed
through standalone iBeamLab settings with equivalent decoded physical samples.

## 4. Complete objective functions and channel selections

Suggested home: `fitting`.

- [x] Implement AutoNRA's Neyman, Pearson, MSE, Poisson, and Gaussian definitions;
  document their exact formulas, floors, and domains.
- [x] Retain iBeamLab's current data-floor and supplied-variance objectives as
  distinct choices.
- [x] Support the required ROI weighting and normalization policies explicitly.
- [x] Define missing-channel, empty-data, overlapping-ROI, and short-prediction
  behavior; preserve AutoNRA behavior through explicit settings where needed.
- [x] Retain per-measurement contributions and invalid-candidate diagnostics.
- [x] Precompute selections and fixed coefficients; score without allocating
  spectrum slices per candidate or ROI.

Completion: identical observations and predictions produce matching AutoNRA
objective values under equivalent settings.

## 5. Compose measurement-specific predictors

Suggested home: `analysis`, reusing existing simulation/inference adapters.

- [x] Bind each measurement to a simulator or a selected surrogate package.
- [x] Combine different predictors within one shared-sample analysis.
- [x] Preserve measurement-specific reference geometry and processing settings.
- [x] Validate which physical settings each predictor permits to vary.
- [x] Group compatible requests into true backend batches and restore output order.
- [x] Retain buffers as populations and sample layer counts change.
- [x] Define ownership, cancellation, and failure propagation across predictors.

Completion: one objective can combine SIMNRA and multiple surrogate packages
without application-specific numerical logic or per-candidate ONNX calls.

## 6. Extend optimizer execution

Suggested home: `fitting`.

- [ ] Expose the optimizer algorithms and settings currently used by AutoNRA.
- [x] Support staged algorithms with per-stage budgets and explicit seed handling.
- [x] Accept multiple initial guesses and carry results between stages.
- [x] Preserve external-optimizer access to the batched objective.
- [x] Report termination reasons, evaluations, invalid candidates, and best results.
- [x] Preserve the best evaluated solution on cancellation without rerunning it.
- [x] Provide scalar progress events and best_result() snapshots on demand.

Completion: AutoNRA's optimizer pipelines can run through iBeamLab with equivalent
settings and no loss of population batching.

## 7. Extend experimental data readers

Suggested home: `io`.

- [ ] Import AutoNRA parser-definition files directly; standalone iBeamLab parser descriptors are implemented.
- [x] Support header markers, data blocks, channel numbering, and count columns.
- [x] Extract real/live acquisition times and convert declared time units.
- [x] Support explicit defaults and missing-channel policies.
- [x] Preserve masks, variances, source information, and useful parse diagnostics.
- [x] Keep instrument file parsing independent of application project parsing.

Completion: representative AutoNRA spectrum files yield equivalent counts and
acquisition settings through C++ and Python readers.

## 8. Add backend diagnostics and scientific export

Suggested homes: `simulation`, `io`, and fitting results.

- [x] Expose available elemental spectra from SIMNRA with labels and units.
- [x] Expose backend capabilities so unsupported diagnostics fail clearly.
- [x] Support inserting measured/predicted spectra into a SIMNRA project and saving
  the fitted state.
- [x] Export fitted profiles, discretized samples, spectra, physical parameters,
  objective breakdowns, and provenance through standalone APIs.
- [x] Make expensive diagnostics optional during fitting.

Completion: AutoNRA can display and export its existing scientific diagnostics
without maintaining a separate SIMNRA implementation.

## 9. Provide parallel analysis execution

Suggested home: `analysis`.

- [x] Provide reusable per-fit workers and explicit CPU/GPU assignments.
- [x] Queue independent measured datasets and preserve input/result association.
- [x] Support initial guesses from previous results or caller-supplied neighbors.
- [x] Keep worker ownership and nested simulator/ONNX thread budgets explicit.
- [x] Support cancellation, partial results, and lightweight progress events.
- [x] Keep worker resources and peak-capacity buffers alive between fits.
- [x] Let AutoNRA choose resource policy; keep execution usable without a GUI.

Completion: many spectra can be fitted concurrently, including mixed CPU/GPU
workers and shrinking DE populations, without duplicate scheduling inside workers.

## 10. Integrate AutoNRA and remove duplicated implementations

- [ ] Add a thin adapter from AutoNRA projects to iBeamLab analysis settings.
- [ ] Replace simulation, objective, parameterization, and optimizer paths in stages.
- [ ] Compare numerical results and representative throughput before each removal.
- [ ] Verify independent C++ and Python execution of the same scientific analysis.
- [ ] Remove each superseded AutoNRA implementation after its replacement passes
  the relevant reference cases.
- [ ] Keep GUI editors, plotting, project persistence, preferences, and runtime
  package installation in AutoNRA.
- [ ] Document the final public APIs and provide small end-to-end examples.

Completion: C++, Python, and AutoNRA run equivalent analyses through one scientific
implementation. AutoNRA contains application integration rather than duplicate
simulation or fitting calculations.
