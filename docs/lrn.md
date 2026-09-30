# Layerwise recurrent network (LRN)

The optional `LRNModel` is a trainable PyTorch forward surrogate built for an
iBeamLab `GenerationStudy`. A shared GRU block processes each layer, sums latent
contributions, and decodes the result into the configured output spectra.

Install the training dependencies with `pip install "ibeamlab[training]"`.
To run the five-layer RBS example against the small generated data, install the
plot extra as well and run `python examples/train_lrn_multilayer.py`.

```python
import torch
import ibeamlab as ibl
from ibeamlab import LRNModel, transforms

study = ibl.GenerationStudy(experiment, parameters, methods=methods)
model = LRNModel(study, {"RBS": 2048}, hidden_size=128)
input_transform = transforms.build_lrn_input_transform(study)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for physical_inputs, target_spectra in training_loader:
    inputs = torch.from_numpy(input_transform.apply(physical_inputs.numpy()))
    prediction = model(inputs)
    loss = torch.nn.functional.mse_loss(prediction, target_spectra)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.export("rbs-lrn.zip", input_transform=input_transform)
```

`build_lrn_input_transform()` first clips and normalizes the concentration
features independently in each layer, then leaves those fractions unchanged
while applying parameter-bound min-max scaling to thickness and setup
features. By default, thickness is mapped from `[0, 100000]` to `[0, 1]`.
The complete pipeline is serialized into the package and applied by native
inference, so callers continue to supply physical parameter values.

The example script loads `layers_01` through `layers_05` with
`ibeamlab.open_dataset`, pads shorter layer systems to the five-layer parameter
layout, trains with an 80/10/10 split, saves model weights and plots, then
exports an iBeamLab package. Override the dataset or training settings with
`--dataset-root`, `--max-layers`, `--epochs`, and `--batch-size`.

`LRNModel` takes input columns in `study.parameters` order. The output column
order follows the experiment detector order, using each detector's length from
`output_spectra_lengths`. The ONNX package records the study's physical sample,
setup, parameter targets, and spectrum labels. Load it in Python with
`ibeamlab.load_model("rbs-lrn.zip")` or use the native C++ `model::ModelPackage`
and `inference::ForwardModel` APIs.

In eager PyTorch inference, a model may accept a shorter number of layers when
the study's parameters are ordered as setup features followed by identical,
contiguous per-layer blocks. ONNX export uses the training schema's fixed input
width; the current native C++ package contract describes a fixed physical
sample layout. The exported graph has a dynamic batch axis.
