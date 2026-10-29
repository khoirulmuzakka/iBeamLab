# Recurrent forward surrogate example

Predict labelled detector spectra from layer thicknesses and elemental
concentrations. The example's `RecurrentSurrogate` architecture uses
scaled elemental areal densities, a shared FiLM-conditioned layer GRU,
summed latent contributions, a spectrum decoder and residual spectral refiner.
It trains one set of weights across the selected layer counts.

The shared GRU processes elemental areal densities from the surface inward.
Independent detector heads decode spectra and refine them locally. The fixed
experimental setup uses the reference's learned constant setup conditioning.
All-zero AD padding preserves state and contributes nothing. Contributions
are summed without additional density weighting; outputs are nonnegative.
An empty stack retains the reference's learned baseline output.
The layer recurrence and defaults follow the reference training workflow.

Model defaults follow the reference training notebook: hidden size 256, layer
embedding 128, setup embedding 32, contribution size 512, recurrent block widths
`[512, 512]`, decoder widths `[512, 512]`, and 32 refiner channels. The configured
refiner kernel is 8, rounded up internally to 9 as in the reference. Both hidden
layer lists remain configurable.

## Run

For an interactive workflow, open [train/train.ipynb](train/train.ipynb).
It includes data inspection, AD distributions, training/resume, held-out
spectrum comparisons and optional package export. Install `matplotlib` and
`jupyterlab` (or `ipykernel`) in your training environment to use the notebook.

`validation.py` defines shared configuration, model, cache and checkpoint checks.
`workflow.load_cache()` is used by the notebook, training and export, so these
entry points apply the same checks. Invalid overrides report the specific field
(for example `training.gradient_clip`), rather than a generic binding/type error.

Install the training dependencies in your active Python environment:

```powershell
python -m pip install torch onnx onnxruntime numpy tqdm
python examples/ibeamnet/run.py prepare --extension-dir build/local/Release/bin
python examples/ibeamnet/run.py train
python examples/ibeamnet/run.py export --extension-dir build/local/Release/bin
```

Edit `surrogate.toml` first. Paths resolve relative to that file. Preparation
requires completed `layers_XX` datasets for all selected `data.layer_counts`.
You may select only completed layer systems while generation continues on others.
For a small initial run, use `prepare --limit 1000`; the limit is per layer system.
Use a separate configuration/cache/run directory for later larger runs.

Training selects CUDA when available. Override with `train --device cpu`.
`train --resume` restores `last.pt`, AdamW and random states. The staged schedule
defines the TOTAL run: 25 epochs at 5e-4, 90 at 1e-4, then 50 at 1e-5 by default.
All stages use the configured batch size (512 by default). Extend or edit
`training.schedule` to continue; completed epochs are skipped and future epochs
use the revised schedule. For example, after 65 epochs, a 165-epoch schedule runs
100 more. Other training settings must match. There is no automatic rate reduction
or early stopping that could skip later stages. `best.pt` is selected by validation
loss across all stages, including the protected pre-training baseline. `last.pt`
is saved before the first update and after every complete epoch, so an interrupted
partial epoch is replayed from its saved start with AdamW and random state restored.
Training reports progress per epoch and writes history and held-out test metrics.
The test split never drives checkpoint selection or learning-rate scheduling.

For a new fine-tuning run, use `train --initial-weights path/to/best.pt`, or set
`INITIAL_WEIGHTS` in the notebook. This loads compatible model weights while
starting a fresh AdamW optimizer and schedule; it does not claim to restore
unavailable optimizer state. Element order, model dimensions, spectrum widths
and scaling must match the training cache. Full training checkpoints include
these fields too, so their best weights can initialize a different run.

The converted reference weights are stored locally at
`checkpoints/reference_rbs/best.pt` (ignored by Git), with verification details in
`conversion.json`. They contain the best model from epoch 160, 13 ordered elements,
2,998 RBS channels, AD factor 1e-4 and count scale 100. The original validation
loss/history are kept as source provenance, not used as metrics for a new dataset.
The notebook selects the checkpoint's architecture, spectrum schema and scaling
when `INITIAL_WEIGHTS` is set. It uses a separate cache for these settings.

For direct PyTorch inference without training or dataset preparation:

```python
from examples.ibeamnet.training import load_checkpoint, make_model
state = load_checkpoint("examples/ibeamnet/checkpoints/reference_rbs/best.pt")
network = make_model(state, state["model_options"]).eval()
network.load_state_dict(state["model_state"])
# network accepts scaled float32 AD [batch,layers,elements].
# Predicted physical counts = network(ad * state["ad_input_factor"]) * 100.
```

Weights-only inference produces the learned spectrum before physical output
corrections. Source inference-correction settings are retained under `state["source"]`.

Export always checks ONNX against PyTorch at batch sizes 1 and 3. By default,
it also verifies physical inference through the C++ API, which requires an
ONNX-enabled extension (see [building](../../docs/building.md)). With your
SIMNRA-only extension, use `export --no-native-check`; Python ONNX verification
still runs, and deployment later requires an ONNX-enabled build.
Export does not modify the extension or interrupt dataset generation.

## Data and scaling

Preparation streams records into float32 NumPy memory maps. Training gathers
whole minibatches; spectra are never loaded into RAM as one dataset. With
2,660,000 records and 5,000 channels, targets alone need about 53.2 GB of disk.
Source datasets remain separate. Cache, run and export directories must be new;
completed manifests are published only after successful preparation/export.

Each layer supplies thickness plus every configured elemental concentration.
Column order is read from dataset metadata. The element set and measurement
setup must agree across source datasets. This example does not train variable
beam energies or detector geometries.

For each element, `AD_e = thickness * atomic_fraction_e`, in units of
1e15 atoms/cm2. The network consumes `AD_e * preprocessing.ad_input_factor`
(default 1e-4), stored directly in the cache. It receives no separate thickness,
fraction or cumulative-depth features. Each detector's counts are divided by
its own `count_scale`.
Short spectra are zero-padded to the configured channel count; nonzero tails
are rejected rather than discarded. Splits are seeded per layer count; all
detector spectra of one sample remain together.

`chi_square` weights squared errors by `1/(target + noise_floor)` in scaled
count units. `mse` is also available. Detector losses are averaged independently
of channel count, with configurable `loss_weight`. Test reports include a
relative L1 error per detector.

## Deployment

Export creates one `forward.ibm` for any positive number of layers. Its ONNX
graph uses a runtime Loop, not a traced fixed-length loop. Batch and layer axes
are dynamic; trailing zero layers are skipped and interior zeros preserve state.
Each format-1 package contains the ordered elements, reference setup, explicit
per-layer AD preprocessing and independent detector postprocessing pipelines.
The C++ runtime converts thickness/fractions to physical elemental AD and applies
the metadata factor once. The graph receives scaled AD. Training layer counts
are provenance only; they never restrict inference. Count scaling is reversed first;
optional exposure scaling and rebinning follow. Enable exposure scaling only
for pileup-free spectra linear in particlesSr. Pileup is not added implicitly.

```python
import ibeamlab as ibl
model = ibl.ForwardModel("examples/ibeamnet/packages/forward.ibm")
spectra = model.predict(experiment)  # physical Experiment with any positive layer count
counts = spectra["RBS"].counts
# Direct physical AD: contiguous float32 [layers,elements], in metadata element order.
# spectra = model.predict_areal_density(ad)
```

To compare a stored sample with a package prediction:

```powershell
python examples/ibeamnet/predict.py examples/ibeamnet/packages/forward.ibm examples/DataGeneration/datasets/iba_multilayer_13el/layers_01 --extension-dir build/local/Release/bin --sample-index 0 --output prediction.npz
```

The NPZ contains `prediction_<label>` and `reference_<label>` arrays. Original
reference spectra may be shorter than the padded model grid.

Pass `--provider cuda --device-id 0` to predict with a CUDA-enabled native build.
For Python integrations, pass `options=ibl.InferenceOptions(execution_provider="cuda")`
to `ForwardModel`; the same options work with inverse models. CPU is the default.

## Verification

`tests/test_forward_surrogate.py` uses temporary dummy datasets to exercise
multi-detector preparation, parameter ordering, split coverage, training,
resume, explicit physical-to-AD conversion, interior/trailing zero-layer
invariance, gradients and dynamic-layer/dynamic-batch ONNX parity.
With an ONNX-enabled extension it additionally checks native inference and
independent detector exposure scaling. GitHub's ONNX integration job runs it.

Caches and checkpoints from the earlier thickness/fraction architecture are
rejected. Prepare a new cache and train a new model for RecurrentSurrogate.
