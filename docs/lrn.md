# Layerwise recurrent network (LRN)

The optional `LRNModel` is a trainable PyTorch forward surrogate built for an
iBeamLab `GenerationStudy`. A shared GRU block processes each layer, sums latent
contributions, and decodes the result into the configured output spectra.

Install the training dependencies with `pip install "ibeamlab[training]"`.
To run the multilayer RBS example, install the plot extra as well and open
`examples/train_lrn_multilayer.ipynb`.

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
features. By default, all layers share a thickness reference equal to the largest declared
thickness upper bound; thickness is divided by that reference, preserving zero.
An explicit `thickness_bounds=(0, reference)` can override it.
The complete pipeline is serialized into the package and applied by native
inference, so callers continue to supply physical parameter values.

The notebook discovers `layers_XX` datasets recursively with
`ibeamlab.open_dataset`, pads shorter layer systems to the configured parameter
layout, trains with an 80/10/10 split, saves model weights, then exports an
iBeamLab package. Edit dataset settings in the dataset cell and batch size and
learning-rate schedule in the training cell.

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
