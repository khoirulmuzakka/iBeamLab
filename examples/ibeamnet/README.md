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
The notebook defines its settings directly in Python and does not read TOML.
It includes data inspection, AD distributions, training/resume, held-out
spectrum comparisons and optional package export. Install `matplotlib` and
`jupyterlab` (or `ipykernel`) in your training environment to use the notebook.
All training-loop controls are beside the training call in section 4: run path,
resume mode, initial weights, staged schedule, batch size, device, workers,
weight decay, gradient clipping, seed, model options and loss. Edit and rerun
that cell without rerunning data preparation or inspection. Its default
`RESUME_TRAINING="auto"` starts a new run when the run directory is absent and
resumes when `last.pt` exists. Explicit `True` requires a checkpoint; existing
directories without one are rejected before any cache access. The data cell's
`SCHEMA_CHECKPOINT` selects cache-compatible dimensions independently of whether
the training cell uses those weights or starts from scratch.

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

For the CLI commands, edit `surrogate.toml` first; notebook users edit the cells.
CLI paths resolve relative to that file. Preparation
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
100 more. Batch size, workers, device, weight decay, gradient clipping and seed
may also change while retaining weights and AdamW moments. Changed loss settings
trigger reevaluation of the protected best under the new objective. Architecture
and cache compatibility are still checked. There is no automatic rate reduction
or early stopping that could skip later stages. `best.pt` is selected by validation
loss across all stages, including the protected pre-training baseline. `last.pt`
is saved before initial validation and after every complete epoch. Startup reports
the loaded weights' validation loss before updates, protected best loss, split
counts, count scales, device and loader settings. Ctrl+C finishes the current
optimizer step and writes both `last.pt` and `interrupted.pt`, including the
partial epoch's shuffle, completed sample count, accumulated loss, optimizer
and RNG states. Resume skips completed batches. A forced interruption inside
an operation saves a diagnostic snapshot and preserves the last safe resume
checkpoint. No held-out testing runs on a graceful interruption.
`prefetch_factor` and `persistent_workers` are editable alongside the training
loop. With workers=0, reads remain synchronous; workers>0 enables these controls.
Optimizer training uses the entire training pool. After each epoch, the frozen
model evaluates a fixed random subset of 20,000 training records and at most
20,000 validation records, using batches of 2,048. Smaller pools use all records.
Subset positions are stored in checkpoints and reused on resume, independently
of training shuffles. `train_loss` is this end-of-epoch evaluation;
`optimization_loss` separately records the online loss during optimizer updates.
An older run's protected best is remeasured if its evaluation policy changes.
Evaluation workers default to zero to avoid extra process pools; optimizer loader
prefetch remains separately configurable. Holdouts contain at most 20,000
validation and 10,000 test records in total; every surplus record returns to
optimizer training. For existing caches, run-local `splits/` indices implement
this without rebuilding spectra. The existing training order remains a prefix,
so a partial epoch retains completed updates and appends newly available rows.
Saved validation evaluation membership is preserved where possible. The final
test uses the complete capped test pool. Checkpoints retain these split indices;
records already used in training never become new holdouts on continuation.
Section 5 displays held-out spectra in a 7-row by 4-column grid (28 panels total).
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
By default, omit `channels` in each `[[spectra]]` section. Preparation scans all
successful records in the selected completed datasets and uses the maximum
stored length independently for each detector. The native scan reuses record
buffers, reads one shard at a time, and creates no Python spectrum arrays.
It adds a sequential read pass before allocating the cache, including records
beyond an optional training-record limit. Resolved widths are saved in the cache,
checkpoints and exported metadata; cache reuse does not repeat the scan.

Short spectra are zero-padded to the inferred width. An explicit positive
`channels` remains available for matching a pretrained checkpoint's output grid;
nonzero tails beyond that width are rejected rather than discarded.
This uses stored array lengths, without trimming trailing zero counts.
Splits are seeded per layer count; all
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
per-layer AD `input_transform`, independent numerical `output_transform`
pipelines and physical detector `output_preparation` pipelines.
The C++ runtime converts thickness/fractions to physical elemental AD and applies
the metadata factor once. The graph receives scaled AD. Training layer counts
are provenance only; they never restrict inference. `predict_reference` and its
batch/direct-AD variants retain numerical transforms and skip physical detector
corrections. Count scaling is reversed first;
rebinning and optional exposure scaling follow. Enable exposure scaling only
for pileup-free spectra linear in particlesSr. Export-only pileup is configured
per detector in `[export.pileup.RBS]`, or in the notebook's `OUTPUT_PREPARATION`
dictionary alongside rebinning and exposure scaling. The notebook previews all
four metadata stages; numerical transform coefficients come from the training
manifest and physical output operations come from these export controls.
The supplied RBS configuration enables it with `tau_seconds=4e-7` and explicitly
declares `targets_pileup_free=true`. Training and cached targets remain bare.
For SIMNRA generation with saved provenance, export verifies the reference
checksum and disabled pileup flag. For other data, the declaration is the
user's statement about the targets. Adding pileup to targets that already
contain it would apply the effect twice.

Processing order is count scaling, rebinning, exposure scaling, then pileup
using requested real/live acquisition times. Each enabled detector expands
from N channels to full `2*N-1` support. These operations are recorded in the
`.ibm` metadata and run in C++, outside the ONNX graph. Native export verification
compares bare network predictions and the corrected pipeline independently.
Load with `options=ibl.InferenceOptions(apply_pileup=False)` to skip declared
forward pileup steps while retaining the other corrections. Export settings
do not invalidate the training cache or require retraining.

```python
import ibeamlab as ibl
model = ibl.ForwardModel("examples/ibeamnet/packages/forward.ibm")
spectra = model.predict(experiment)  # physical Experiment with any positive layer count
counts = spectra["RBS"].counts
# Direct physical AD: contiguous float32 [layers,elements], in metadata element order.
# spectra = model.predict_areal_density(ad)
```

For parallel fitting, assign one model instance to each measured spectrum's
worker. Evaluate each differential-evolution population with one ONNX call:

```python
import numpy as np
# Physical elemental AD [population,layers,elements], in packaged element order.
ad = np.ascontiguousarray(candidate_areal_densities, dtype=np.float32)
# Use the sum of processed detector widths (2998 for the converted RBS package).
out = np.empty((len(ad), 2998), dtype=np.float64)
predictions = model.predict_areal_density_batch(ad, out=out)
population_counts = predictions["RBS"]
```

Reuse `out` across iterations. Returned arrays view that storage, so reusing it
overwrites prior predictions. Without `out`, each call owns its output. Each
worker may select CPU or CUDA independently; calls on one model must serialize.
The experiment-based `predict_batch()` also uses one ONNX call, with zero padding
for differing layer counts. Detector postprocessing remains independent.

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
