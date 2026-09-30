# Model package version 2

An iBeamLab model package is a directory or ZIP containing `package.toml` and
`model.onnx`. The root explicitly declares `model_type = "inverse"` or
`model_type = "forward"`. A package cannot serve both directions.

Common ONNX tensor names, dimensions, opset, payload size, and CRC32 are stored
in `[model]`. `[transforms.input]` is applied before ONNX execution and
`[transforms.output]` is inverted afterward.

Input transforms may be pipelines. In addition to ordinary affine transforms,
the native runtime supports layerwise concentration normalization and
parameter-bound min-max scaling. Concentration-group column indices are stored
in the manifest so the same per-layer normalization used for training is
reproduced during inference.

Every transform is validated recursively when a package is created or opened.
Dimensions, numeric parameters, concentration-column groups, and child
pipelines must be consistent with the ONNX model metadata; unsupported or
malformed transform specifications are rejected before an inference session is
created.

An inverse package contains sample/setup templates, ordered input spectrum
labels and lengths, and ordered output parameters with physical targets,
bounds, and units. `InverseModel` executes spectra-to-sample inference.

A forward package contains sample/setup templates, ordered input parameters
with physical targets and bounds, and ordered output spectrum labels and
lengths. `ForwardModel` executes sample/setup-to-spectra inference. It does not
implement `ISimulator` and is not a backend for `DataGenerator`.

The C++ writer calculates payload size and CRC32, refuses to overwrite existing
outputs, and writes directory and ZIP packages. Python exposes the same API.

For the trainable PyTorch layerwise recurrent forward model and package export
workflow, see [LRN](lrn.md).
