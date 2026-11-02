# Smooth profiles and sublayers

`profile` adds smooth composition functions without changing the sample consumed
by simulators or surrogate models. Macro layers discretize into ordinary
homogeneous `Sample` layers ordered from surface to depth.

```python
import ibeamlab as ibl

profile = ibl.ProfileModel((
    ibl.ProfileLayer(
        ibl.Layer(1000, {"Si": 0.5, "O": 0.5}),
        (ibl.polynomial(1, -1), ibl.polynomial(0, 1)),
        sublayers=20,
    ),
))
sample = profile.discretize()
```

Functions follow reference species order and describe composition weights.
Each midpoint's absolute function values are normalized to sum to one. A zero
or nonfinite total is an error, not an implicit composition assumption.
Isotope mixtures, roughness and porosity are copied from the macro layer.

For local normalized depth `y`, the functions are:

| Function | Weight |
| --- | --- |
| `constant(v)` | `abs(v)` |
| `polynomial(c0, c1, ...)` | `abs(c0 + c1*y + ...)` |
| `erf_polynomial(A, a, b, c, p1, ...)` | `abs(A*(1+a*erf((y-b)/c))*(1+p1*y+...))` |

The erf width `c` must be strictly positive. Physical function evaluation takes
`shift` and `width` in `1e15 atoms/cm2`, returning zero outside the layer.
Linear grids use equal-width sublayers. Logarithmic grids use base-4 edges with
exponents from -3 to 0, shift the first edge to zero, and restore the final edge
to the full layer thickness. Both evaluate composition at sublayer centers.

## Fit coefficients

```python
problem = ibl.FitProblem(
    data=measured_data,
    initial_sample=profile.reference_sample(),
    parameters=[ibl.vary.layer_thickness(layer=0, bounds=(500, 1500))],
    predictor=simulator_or_surrogate,
    profile=profile,
    profile_parameters=[ibl.ProfileParameter(
        layer=0, species=0, coefficient=1, bounds=(-1, 0), name="silicon_slope")],
)
result = ibl.fit(problem, max_evaluations=1000)
```

With a profile, ordinary sample targets address **macro layers**. Coefficient
indices are zero-based in the function's coefficient list. Fixed coefficient
bounds are supported. Optional `links=((layer, species, coefficient, ratio), ...)`
on a ProfileParameter ties additional coefficients to a ratio of its value;
initial values must agree. Do not combine profile weights with direct fraction or AD
parameters: the functions define the sublayer fractions. The predictor must
support changing the discretized sample. The result contains physical coefficient
values and the evaluated discretized sample/prediction, not a regenerated fit.

See [the runnable example](../examples/Fitting/fit_profile.py).

## C++ and storage reuse

Public headers are `profile/functions.h`, `profile/model.h`, and
`profile/discretization.h` under `ibeamlab/`.

```cpp
ibeamlab::profile::Discretizer grid(model);
ibeamlab::core::Sample output;
grid.apply(model, output);
// Change coefficients or layer thickness, then reuse the same output.
grid.apply(model, output);
```

Grid edges are compiled once. Sublayer counts, species counts and spacing must
remain unchanged. Each call reuses existing layer/species storage at the same
topology. Inputs and output must not alias. The fitting objective retains macro
and discretized storage separately, avoiding reconstruction on every candidate.
Each discretizer/objective belongs to one caller.

Python `Discretizer.apply()` returns an independent owning Python sample;
`apply_into(model, native_sample)` avoids that output copy for low-level usage.
