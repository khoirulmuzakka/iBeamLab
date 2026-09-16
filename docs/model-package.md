# Model package version 1

The package extension is `.ibeam.zip`. A package is a ZIP archive containing
`package.toml` and `model.onnx`. Directory packages with the same files are
supported for development.

`package.toml` starts with:

```toml
format = "ibeamlab.onnx-package"
format_version = 1
methods = ["RBS"]
spectrum_lengths = [2]
input_features = ["RBS:0", "RBS:1"]
output_features = ["parameter"]
output_units = ["1e15 atoms/cm2"]

[model]
onnx_file = "model.onnx"
input_name = "inputs"
output_name = "outputs"
input_dimension = 2
output_dimension = 1
opset_version = 18
size = 1755
crc32 = "03b53033"
```

`[preprocessing.input]` and `[preprocessing.output]` describe `identity`,
`constant_factor`, `standard_scaler`, `min_max_scaler`, `log`, or `pipeline` transforms.
Pipelines contain an array of child transform tables. Unknown optional TOML keys
are ignored; unknown format versions and transform types are rejected.

The reader never extracts ZIP paths. It reads only the exact `package.toml` and
declared root-level ONNX entry into bounded memory and lets miniz validate ZIP
CRCs. Declared `size` and `crc32` values are checked against the ONNX payload.
Method order and `spectrum_lengths` define concatenation order.
