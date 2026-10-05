# IBAnet inverse training

Install the optional dependencies with `pip install -e ".[training,plot]"` and
open [train_ibanet.ipynb](../examples/train_ibanet.ipynb). Set `DATASET_ROOT`,
`CONFIG_PATH`, `MAX_LAYERS`, and `OUTPUT_DIR` before running it. The notebook
expects complete `layers_XX` datasets from a fixed measurement setup. It checks
their generation metadata against the selected configuration.

`from ibeamlab import IBAnet, IBAnetLoss` loads the optional PyTorch module.
IBAnet takes a `GenerationStudy` and a mapping of detector labels to channel
counts. Every detector has a separate strided 1D CNN. Their features are
concatenated and passed to an MLP with mean, standard-deviation, and presence
heads. Detector order follows the study; element order is explicit or follows
the first template layer. Every template layer must contain those elements.

```python
model = IBAnet(study, {"RBS": 2048, "PIXE": 1024})
prediction = model(raw_concatenated_spectra)
loss = IBAnetLoss()(prediction, scaled_edp_targets)
loss.backward()

probability = prediction.P
conditional_mean = prediction.Y
conditional_std = prediction.R
posterior_mean = prediction.mean
posterior_variance = prediction.variance
draws = prediction.sample(count=16)
```

All prediction matrices have shape `[batch, max_layers, elements]`; draws have
shape `[draw, batch, max_layers, elements]`. Input spectra are raw finite,
nonnegative counts. `log1p` and per-channel standardization run inside the model.
Fit `input_mean` and `input_std` on the training split's log counts and supply
them to the constructor. They are saved as buffers in the state dict and ONNX.

The targets are `layer_thickness * elemental_concentration` in each cell.
Unoccupied deeper layers are zero padded. The notebook multiplies densities by
`TARGET_SCALE`. Both `Y` and `R` use these scaled units; divide by that factor
for physical units and divide variance by its square.

The loss implements the spike-and-slab likelihood in [draft.tex](draft.tex):
binary cross entropy on presence logits plus a Gaussian NLL only for positive
target entries. It sums cells per sample and averages samples. It does not apply
regression to absent cells. `binary_cross_entropy_with_logits(q, M)` evaluates
`-M*log(P) - (1-M)*log(1-P)` with `P = sigmoid(q)` in a numerically stable form.
The training loss omits `0.5*log(2*pi)` per present cell: it depends only on the
known target mask, so removing it changes loss values but not gradients or the
optimum. Reported notebook NLL values omit this constant as well.
Gaussian means use softplus, restricting the draft's
mean to positive values; standard deviations use softplus plus a positive floor.
The Gaussian distribution remains untruncated, so raw draws can be negative.
Projection or rejection for physical optimizer initialization must be explicit
and changes that distribution. Independent cells do not impose a contiguous
layer mask or correlations between compositions.

`model.export(path, output_inverse_factor=TARGET_SCALE)` writes a native format
version 3 package whose point estimate is `P * Y`. Input preprocessing is in the
ONNX graph; the package's output transform reverses target scaling. The package
contains the reference sample/setup, detector lengths, element order, and EDP
layout. It requires all declared spectra under the reference conditions.
Exports default to `need_pileup_subtraction=True`, written as
`inverse.need_pileup_subtraction = true` in package metadata, because the current
datasets were generated with pileup disabled. The package also stores
`pileup_fudge_factor_seconds` (default `0.4e-6`, configurable during export).
Native `predict()` uses this coefficient with the experimental acquisition times
for pileup removal, rebins to reference calibration, and scales exposure before
applying the training transforms. Both reference and measured ParticlesSr must
be positive. The complete pileup tail and reference energy-range coverage are
required. Unsupported beam/resolution differences warn and are not corrected.
For a model trained with pileup present, explicitly export with
`need_pileup_subtraction=False` to skip subtraction; calibration/exposure
corrections then do not account for nonlinear changes to existing pileup.

```python
from ibeamlab import InverseInput, InverseModel

model.export("ibanet.zip", output_inverse_factor=1e-3)
deployed = InverseModel("ibanet.zip")
measurement = InverseInput(
    {"RBS": rbs_spectrum, "PIXE": pixe_spectrum}, experimental_detectors,
)
result = deployed.predict(measurement)
physical_edp = result.values
```

For already pileup-free experimental spectra, pass
`pileup_already_removed=True` to `InverseInput` (the dataclass is
frozen): this bypasses subtraction while retaining rebinning and exposure
corrections. For dataset spectra already at training conditions use
`deployed.predict_prepared(spectra)`; embedded input normalization still runs.

The native package exposes the posterior mean, which can retain small positive
densities in absent elements and padded layers. It does not expose `Y`, `R`, or
`P`. Keep the PyTorch checkpoint for probabilistic predictions and sampling.
The notebook saves architecture, layout, target scale, configuration, split
indices, normalization buffers, and best validation weights together. Its
checkpoint reload option fine-tunes the best weights with a fresh optimizer.

The notebook evaluates held-out NLL, physical posterior-mean MAE, presence Brier
score, and conditional Gaussian coverage. Its final cell compares batched native
ONNX predictions with PyTorch posterior means. It loads data eagerly; use a
streaming training dataset for collections that exceed available memory.
