# Native architecture

The public C++ API uses the `ibeamlab` namespace and C++20. Physical units are:
beam energy, spread and detector resolution in keV; layer thickness in
1e15 atoms/cm2; acquisition times in seconds; isotope mass in atomic mass units.

The single `ibeamlab` library contains these concrete responsibilities:

- sample structures and experimental setup;
- simulation requests/results and SIMNRA execution;
- parameter materialization and batch generation;
- streaming, sharded dataset storage;
- rebinning, pileup, and reversible transforms;
- safe TOML/ZIP model-package loading;
- long-lived ONNX Runtime inference sessions.

Consumers link only `ibeamlab`; the responsibilities above are C++ namespaces,
not separately deployed libraries.

SIMNRA workers are persistent. Each worker constructs, uses, and destroys its COM
objects on the same worker thread. Cancellations are cooperative. Batch results
retain request order and may contain structured per-item failures.

Public containers preserve insertion order. Concentrations and isotope fractions
must sum to one within an absolute tolerance of 1e-6; they are never silently
normalized. A missing isotope list means natural abundance. Spectrum counts use
32-bit floats for storage and ML interoperability; physical parameters use doubles.
Sample models and experimental setups round-trip through versioned TOML using
`sample::toToml`, `sampleModelFromToml`, and `experimentalSetupFromToml`.

`compile.bat Release` (or `Debug`) places a flat runtime bundle in `lib`; no
configuration-named subdirectory is created there.

ONNX inference defaults to one intra-op and one inter-op CPU thread. Graph
optimizations are opt-in through `InferenceOptions::enableGraphOptimizations`;
this keeps the default deterministic and avoids oversubscribing generation
workers. Applications may tune these values after benchmarking their model.
