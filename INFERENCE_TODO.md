# iBeamLab ONNX inference redesign

The physical simulator used by `DataGenerator` and trained ONNX models are
separate systems. `SimnraSimulator` remains a data-generation backend.
`ForwardModel` and `InverseModel` are deployment inference APIs and do not
implement `ISimulator`.

## Implementation checklist

- [x] Move physical parameter targets and read/write operations into a neutral
  parameter schema shared by generation and inference.
- [x] Extract raw ONNX matrix execution into a private reusable session.
- [x] Replace ambiguous package metadata with explicit forward/inverse model
  metadata.
- [x] Implement `ForwardModel`: sample/setup to labeled spectra.
- [x] Implement `InverseModel`: labeled spectra to a materialized sample model.
- [x] Keep SIMNRA and `DataGenerator` independent of ONNX inference.
- [x] Serialize and validate both package types using TOML and ZIP packages.
- [x] Expose forward and inverse APIs through pybind11 and transparent Python
  modules.
- [x] Add native numerical tests for both inference directions.
- [x] Add an independent C++ consumer example for the forward API.
- [x] Update package and architecture documentation.
