# Dataset format version 1

An iBeamLab dataset is a directory containing `dataset.toml`, `failures.ibd`, and
one or more numbered `part-XXXXXX.ibd` shards. The TOML file identifies the format
as `ibeamlab.dataset`, records completion state and provenance, and lists shards,
parameters, and spectrum labels. Provenance includes build/platform, simulator,
sampler name/version, and seed.

Shard files begin with the eight-byte magic `IBEAMDS1`. Records contain a sample
index, UTF-8 identifier, float64 open parameters, and labeled float32 spectra.
Strings and collection sizes are length-prefixed. `failures.ibd` uses the same
magic and stores the sample index, identifier, method label, error type, and message.

Raw shard records retain each spectrum's original channel count. During streaming
generation, the writer tracks the maximum count independently for every spectrum
label and publishes those maxima as `spectrum_lengths` when finalizing the manifest.
Readers validate the recorded lengths and zero-pad the high-channel end of shorter
spectra to the declared per-label maximum. This produces rectangular ML inputs
without cropping simulator output or buffering the complete dataset in memory.

The manifest embeds four reproducibility sections. `generation` contains the
complete baseline sample and experimental setup, ordered methods, parameter
targets, bounds, and units. `generation_options` records batch/shard settings,
seed, failure policy, and the external sampler identity. `sampling_config`
contains the Python-defined sampling policy as TOML; its exact fields are owned
by that sampler rather than the native dataset format.
`simulator_config` records the concrete backend and its settings; SIMNRA entries
include worker/apartment/calculation settings plus each reference file's absolute
path, byte size, and CRC32 checksum.

Writers create an incomplete manifest immediately. Manifest updates use a temporary
file followed by rename. Only successful finalization marks it complete. Readers
reject unknown format versions, malformed magic,
truncated records, and values exceeding defensive size limits.
