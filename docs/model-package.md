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


Inverse EDP format version 4 adds two named ONNX float32 outputs alongside the
posterior mean. All outputs are `[batch, max_layers * number_of_elements]` in
layer-major order. The inverse table requires `presence_probability_output_name`,
`posterior_std_output_name`, `posterior_std_inverse_factor` (finite and positive),
and `uncertainty_predicted` (boolean). The runtime applies the existing output
inverse transform to the mean, leaves probabilities unchanged, and divides
standard deviations by the independent std factor. Standard deviations must
already describe the full posterior, including presence uncertainty.

Every `EdpMap` has matching `values`, `presenceProbability`, and `posteriorStd`
matrices (Python: `values`, `presence_probability`, `posterior_std`). Validation
requires finite nonnegative means/std and probabilities in [0, 1]. Deterministic
models may emit probability 1 and std 0 and set `uncertainty_predicted=false`.
Version 3 inverse packages remain supported with deterministic defaults; version
2 forward packages are unchanged. Adding fields changes the C++ ABI: rebuild
native consumers and restart Python kernels after updating the extension.
