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
matching metadata encoder/decoder. Format changes require a new version tag.

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

A `.ibm` envelope uses magic `IBMODEL1`; only format 1 is supported. Its payload
starts with direction (uint64: forward=0, inverse=1) followed by exactly one
schema and its ordered pipelines:

- Forward: reference experiment, ordered input parameter targets/units, spectra
  schema, tensor preprocessing, per-spectrum postprocessing.
- Inverse: reference measurement setup, spectra schema, EDP element names/layers/
  uncertainty, per-spectrum input preparation, tensor preprocessing, EDP scaling.

The spectra schema is a count followed by label/channel-count pairs. Each graph
spectrum occupies a contiguous slice in that declared order, irrespective of the
caller's container order. Labels must be unique and resolve to reference
measurements. Their channel counts may differ. The schema describes graph
channels; physical postprocessing such as pileup may extend the returned spectrum.

A tensor pipeline is an ordered list of tagged steps:

| Tag | Operation | Encoded parameters |
| --- | --- | --- |
| 0 | Log1p | None |
| 1 | MinMaxScale | minimum/maximum arrays, low/high range, clip flag |
| 2 | StandardScale | mean and deviation arrays |
| 3 | MultiplyScale | factors array |
| 4 | CustomTransform | identifier, implementation version, named numeric arrays |

A spectrum pipeline contains labelled entries, each with an ordered list of:

| Tag | Operation | Encoded parameters |
| --- | --- | --- |
| 0 | MultiplyCounts | factor |
| 1 | ExposureScale | None; reference/requested ratio follows model direction |
| 2 | Rebin | destination channel count, require-full-coverage flag |
| 3 | Pileup | tau seconds, acquisition source |
| 4 | RemovePileup | tau seconds, relative tolerance, acquisition source |

Acquisition source is requested/measured=0 or reference=1. EDP postprocessing is
an ordered list of positive scaling factors applied to means and std only.
Unknown operation tags are rejected. Unavailable custom implementations fail at
inference model construction; package inspection does not require the plugin.
All pipeline/list lengths and array sizes are bounded before decoding allocations.

Input tensor name, output tensor name and provenance follow the active schema
and pipelines. Finally the payload contains graph byte length and embedded ONNX
bytes. Shared/direction-specific metadata is declared in [model/metadata.h](../cpp/include/ibeamlab/model/metadata.h);
[model/package.h](../cpp/include/ibeamlab/model/package.h) contains the package I/O interface.

Graph loading verifies one rank-2 float32 input/output, names and feature widths.
Inputs/outputs use `[batch,width]`, with graph spectrum slices defined above.
Fitted transformations run outside ONNX in their recorded order and are never
refit at inference. Physical outputs must be finite/nonnegative and probabilities
must remain in [0,1]. Standard deviations use the same physical scale as means.

Writes reject existing destinations. Callers exclusively own output paths;
concurrent writers to the same destination are unsupported.
