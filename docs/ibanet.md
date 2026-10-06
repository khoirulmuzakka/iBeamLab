# IBAnet inverse training

Install the optional dependencies with `pip install -e ".[training,plot]"` and
open [train_ibanet.ipynb](../examples/train_ibanet.ipynb). Set `DATASET_ROOT`,
`CONFIG_PATH`, `MAX_LAYERS`, and `OUTPUT_DIR` before running it. The notebook
expects complete `layers_XX` datasets from a fixed measurement setup. It checks
their generation metadata against the selected configuration.

`from ibeamlab import IBAnet, IBAnetLoss` loads the optional PyTorch module.
IBAnet takes a `GenerationStudy` and a mapping of detector labels to channel
counts. Every detector has a separate residual strided 1D CNN with GroupNorm.
Flattened features retain absolute energy position and are fused into a
LayerNorm-normalized spectrum context. A unidirectional GRU decodes layers
from the surface toward the bottom, with spectrum context and a learned
layer-position embedding at every step. Its initial hidden state also comes
from spectrum context. Shared mean, standard-deviation, and presence heads
operate on each layer state. There is one Gaussian slab per cell, without
mixture components, teacher forcing, or feedback of sampled predictions.
Detector order follows the study; element order is explicit or follows
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

Defaults are `cnn_channels=(32, 64, 128, 128)`, `kernel_size=7`,
`decoder_hidden_size=256`, `decoder_layers=2`, `layer_embedding_size=32`,
`head_hidden_sizes=(128,)`, and `dropout=0.1`. Dropout is applied between GRU
layers and in the shared output MLP; it is disabled by `predict()` and export.
`head_hidden_sizes` configures the shared MLP **after** the GRU. GroupNorm and
LayerNorm use no running batch statistics. The constructor's `architecture`
dictionary contains all network options needed to reconstruct a checkpoint.

The architecture replaces the former CNN/MLP, so its PyTorch weights cannot
be loaded into the new model. Retrain into a new output directory. Existing
exported ONNX packages remain usable with the inference API.

All prediction matrices have shape `[batch, max_layers, elements]`; draws have
shape `[draw, batch, max_layers, elements]`. Input spectra are raw finite,
nonnegative counts. `log1p` and per-channel standardization run inside the model.
Fit `input_mean` and `input_std` on the training split's log counts and supply
them to the constructor. They are saved as buffers in the state dict and ONNX.

The targets are `layer_thickness * elemental_concentration` in each cell.
Unoccupied deeper layers are zero padded. The notebook multiplies densities by
`TARGET_SCALE`. Both `Y` and `R` use these scaled units; divide by that factor
for physical units and divide variance by its square.
Use one global target scale, with typical positive entries preferably around
0.1–1; keep it fixed across NLL comparisons. The notebook retains `1e-5` for
comparison with earlier runs. Scaling does not normalize each sample's total.

The notebook enables `POISSON_NOISE=True` for datasets of noise-free expected
counts. Each raw training batch receives a fresh `torch.poisson` draw before
the model's embedded log1p and standardization. Targets are not perturbed.
Validation and test spectra receive fixed CPU draws with seeds `SEED + 1` and
`SEED + 2`; those same test inputs are used for metrics and native export parity.
Normalization is still fitted on clean training expected counts. Disable the
option for already noisy data and rerun the split/build section when changing
it. Inference does not add noise to measured counts.

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
and changes that distribution. Recurrent states make deeper parameters depend
on earlier layer states, but the likelihood and samples still factorize over
cells given the spectrum. They do not impose a contiguous layer mask or joint
covariance between compositions.

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
checkpoint reload option is enabled by default: rerunning the training cell
after interruption loads compatible best weights automatically. It fine-tunes
with a fresh optimizer and restarts the configured schedule, rather than
resuming the exact interrupted epoch. For a fresh run, rebuild the model and
select a new output directory. Incompatible checkpoints are still rejected.
Noise settings are recorded in checkpoint signatures. When only the noise
policy changes, compatible weights can be reused; the notebook clears the old
loss history and recomputes baseline validation NLL on the current fixed inputs.

The notebook evaluates held-out NLL, physical posterior-mean MAE, presence Brier
score, and conditional Gaussian coverage. Its final cell compares batched native
ONNX predictions with PyTorch posterior means. It loads data eagerly; use a
streaming training dataset for collections that exceed available memory.
