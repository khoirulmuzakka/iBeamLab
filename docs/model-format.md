# iBeamLab format 1

Datasets and model packages use binary format 1. Unknown magic/version tags, invalid lengths, trailing fields and checksum
failures are rejected. Files are limited to 512 MiB payload each. Readers bound
nested counts before allocating. Checksums detect accidental corruption; they
are not signatures.

Every file has an 8-byte ASCII magic, a little-endian uint64 payload length,
a uint64 FNV-1a checksum of the payload, and that payload. Numeric scalar fields
use uint64 or IEEE-754 binary64 in little-endian order. Text is byte-length
prefixed UTF-8; arrays are count-prefixed. ONNX graph bytes are opaque. Current
field order is defined by the encoder/decoder in [cpp/src/io.h](../cpp/src/io.h) and each module's
matching metadata encoder/decoder. This format-1 schema replaces earlier development
layouts; compatibility with those layouts is not provided.

## Datasets

A dataset is a newly created directory. `incomplete.ibd` (magic `IBDATA01`)
records its initial schema. Shards `shard-N.ibd` (magic `IBSHRD01`) are published
by renaming complete temporary files. A committed `manifest.ibd` (magic
`IBDATA01`) contains the final schema, counters and shard count. Only its
presence signifies completion. Readers reject datasets without it. Publication
is atomic visibility, not a power-loss durability guarantee; files are not fsynced.

Metadata stores the reference experiment, ordered named parameter targets/units,
provenance, and requested/accepted/failed/discarded counts. A record stores its
increasing sample index, nonempty ID, physical parameter row and either ordered
labeled spectra or a failure with measurement/message. Failed records may have
physically invalid finite parameter rows. Nonfinite rows are rejected.
Spectrum lengths may vary between records; model consumers must reconcile them
explicitly. Readers hold one bounded shard and reuse a caller's record buffers.

## Model packages

A `.ibm` envelope uses magic `IBMODEL1`; the scientific package format remains
**1**. Its payload begins with the text `ibeamlab.model` and uint64 `1`,
then direction (forward=0, inverse=1). Previous development layouts are rejected;
export existing weights again with the staged contract. Experimental setup encoding includes explicit detector geometry.

The logical execution order is:

```text
physical input -> input_preparation -> input_transform -> ONNX
               -> output_transform -> output_preparation -> physical result
```

Forward metadata encodes reference setup, input-preparation schema tag/fields,
reference spectra, numerical input transform, per-detector numerical output
transforms, and per-detector physical output preparation. Input tag 0 stores a
reference sample and parameter targets/units; tag 1 stores ordered elements and
explicit isotope fractions for each element,
without a sample or layer limit. Packing computes physical elemental AD from
thickness/fractions; direct AD avoids sample construction.

Inverse metadata encodes reference setup, reference input spectra, EDP output
schema, physical input preparation, per-detector numerical input transforms, and
numerical output transform. The EDP schema contains ordered elements, fixed
maximum layers, uncertainty flag and unit `1e15 atoms/cm2`.

Each reference spectrum stores measurement label, channel count, pileup state
(bare=0, piled-up=1) and unit `counts`. The matching reference measurement supplies
calibration coefficients, beam conditions, particlesSr and real/live acquisition
times. The channel count plus calibration defines its reference energy grid.

### Numerical training transforms

`TransformPipeline` stores direction (forward=0, inverse=1), execution location
(runtime=0, graph=1), then an ordered list of numerical operations:

| Tag | Operation | Stored coefficients |
| --- | --- | --- |
| 0 | Log1p | None |
| 1 | MinMaxScale | minimum/maximum arrays, low/high range, clip flag |
| 2 | StandardScale | mean and deviation arrays |
| 3 | MultiplyScale | factors array |
| 4 | CustomTransform | identifier, implementation version, named numeric arrays |

Inverse direction reverses operation order and applies each inverse. Scalars
broadcast; arrays match feature width. Clipped min-max scaling and zero
multipliers cannot be inverted. Fitted coefficients are never refitted at load
or inference. A labelled `SpectrumTransform` associates a pipeline with one
detector, allowing independent input/output scalers and execution locations.

Runtime transforms execute outside ONNX; Graph transforms declare computation
already embedded in the exported graph and are skipped by the runtime. Exporters
must declare placement honestly: an opaque graph cannot prove that it contains
a particular training scaler. Raw graph parity and full reference/physical parity
are separate checks. The supplied example exports AD/count scaling at Runtime.

Inverse uncertainty packs graph means, presence probabilities and posterior
standard deviations. Affine output transforms shift/scale means and multiply
standard deviations by absolute scale. Probabilities remain unchanged. Nonlinear,
custom or clipped uncertainty transforms are rejected, rather than applying a
nonlinear function to a standard deviation.

### Physical measurement preparation

Physical pipelines contain labelled entries with ordered operations:

| Tag | Operation | Encoded parameters |
| --- | --- | --- |
| 0 | ExposureScale | Reference/requested ratio follows model direction |
| 1 | Rebin | Destination channel count and coverage policy |
| 2 | Pileup | Resolving time in seconds and acquisition source |
| 3 | RemovePileup | Resolving time, tolerance and acquisition source |

Acquisition source is requested/measured=0 or reference=1. Constant count scaling
belongs to numerical output transforms, not physical preparation. Operation order
is explicit; pileup and exposure scaling generally cannot be rearranged. Reference
pileup states are validated to reject repeated pileup or inconsistent preparation.
Pileup expands N channels to full `2*N-1` support; removal needs that full support.

`predict` executes all applicable stages. `predict_reference` bypasses physical
measurement preparation while retaining numerical transforms and physical output
decoding. Both have batch paths; direct AD population APIs support caller-owned
packed float64 outputs. `native.run_graph` bypasses every runtime stage.

### Loading and allocation

Model loading validates shapes, units, labels, coefficient widths, directions,
placement and supported operations, then resolves kernels and detector slices.
Custom Runtime kernels must be registered in C++ before loading; inverse custom
operations require an inverse kernel. Graph custom operations need no runtime
transform registration. Pipelines are compiled once and fitted arrays stay owned
by immutable model metadata. Workspaces and ONNX tensor handles are retained.

Parameter forward and inverse graphs use rank-2 `[batch,width]` inputs; layer
forward graphs use rank-3 `[batch,layers,elements]` with dynamic batch/layer axes.
Outputs are packed rank-2 float32 tensors. Logical transforms preserve width.
Layer Runtime transforms skip empty rows; a transform that turns a nonempty
layer into the all-zero padding sentinel is rejected. Empty/Graph input stages
borrow direct contiguous float32 AD without an input copy. Caller-owned results
avoid additional spectrum storage allocations after warmup.

Input/output tensor names, provenance and embedded standalone ONNX bytes follow
the directional schema. External ONNX tensor files are unsupported. All counts
and array sizes are bounded before decoding allocations. See
[metadata.h](../cpp/include/ibeamlab/model/metadata.h) and
[package.cpp](../cpp/src/model/package.cpp) for the exact field encoding.

### Isotope assumptions

Samples use an empty isotope list for iBeamLab's natural composition. An explicit
entry is identified by element symbol and positive mass number; exact masses come
from `core/isotopes.h`. Fractions are finite, nonnegative, unique by mass number,
and sum to one. Elements without a defined natural mixture require explicit
isotopes. The initial database is the mass/abundance snapshot used by AutoNRA,
covering supported isotopes of H through Es; it is a fixed convention rather than
a sample-specific measurement of natural variation.

Layer input schemas accept optional `isotope_compositions` keyed by element.
Omitted assumptions resolve to the framework natural mixture during construction.
Export writes every assumption explicitly; loading never substitutes new natural
fractions for stored fractions. Fixed-parameter packages likewise freeze their
reference sample's compositions. The package contract remains format 1; packages
from earlier development layouts must be exported again from their weights.

Inference compares isotope identities and fractions independently of ordering.
Explicit zero-fraction entries do not change composition. Elemental AD inputs use
the stored mixtures and cannot express varying enrichment. A different requested
mixture is rejected with the element name. Direct AD tensors carry no isotope
information and are interpreted using the package assumptions.

Dataset sample encoding retains its existing field layout, including a reserved
mass slot, but newly written samples store resolved isotope lists. This records
the generation convention. SIMNRA receives the same resolved fractions, with no
per-isotope COM updates when the mixture is unchanged. Existing layer replacement
continues to append new layers before removing old layers.

### Geometry in the format-1 setup contract

Each serialized experimental setup starts with `ibeamlab.setup.geometry` and
uint64 `1`. Every detector includes incidence, exit and scattering angles in
degrees after its information string. The same setup encoding is shared by
model packages and dataset metadata. Parameter tags 9, 10 and 11 identify
incidence, exit and scattering angles, respectively, selected by measurement label.

Earlier development metadata without geometry is rejected. Re-export model
packages with actual geometry. Existing dataset metadata needs to be regenerated
from its true experimental reference; missing angles must not be guessed from a
default. Existing training-cache reference dictionaries without geometry are
rejected on export for the same reason. Formats remain 1.
