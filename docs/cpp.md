# C++ API and scientific contracts

Build and install through the root project; see [building.md](building.md).
Public headers live under cpp/include/ibeamlab and document units and ownership.
Consumers link the exported target ibeamlab::ibeamlab.

## Modules

- `core`: shared samples, independently configured measurements, parameter targets,
  validation, units, owning spectra and non-owning views.
- `processing`: calibrated rebinning and pair-sum pileup application/removal.
- `simulation`: deterministic dummy spectra and persistent worker-owned SIMNRA COM sessions.
- `datasets`: checksummed sharded storage, transactional completion and streaming cursors.
- `generation`: physical parameter matrices, uniform sampling, failure policies and progress.
- `model`: shared package schema and embedded ONNX payload.
- `inference`: forward experiment-to-spectra and inverse spectra-to-EDP APIs.

All modules currently compile into one library. Public headers have no ONNX or
SIMNRA dependency. Model instances and dataset cursors are owned by one caller;
use separate instances per worker. The SIMNRA batch method uses its configured
worker pool and preserves input/measurement order. Target rebuilds append and populate
replacement layers before removing the original prefix, preserving cross-section
choices while elements remain present. Matching topology is normally reused;
`SimnraConfig.reuse_target_topology=false` forces rebuilding for diagnostics.
Pileup follows the reference project setting; `nopu` references are checked to
have it disabled. Cancellation is cooperative
between measurements; an active external COM calculation cannot be interrupted.

## Scientific and allocation contracts

Counts use contiguous `double` storage and represent integrated counts per
channel, not count density. ONNX tensors use float32. Calibration describes
channel centers `E(i)=offset+linear*i+quadratic*i*i`; edges are evaluated at half
integers. All edges must be finite and strictly increasing. Rebinning assumes
uniform density inside each source bin. It conserves counts over common support;
counts outside the destination support are discarded without renormalization.
`RequireFull` rejects a destination grid not covered by the source.

Rebinning into caller-owned output performs no allocations. Pileup application
uses a reusable FFT workspace and allocates only when its capacity grows.
The benchmark checks allocation counts after warmup at 5,000 channels. Inverse
pileup currently uses a quadratic recurrence: optimize it only after profiling.
Output buffers must not overlap input buffers. C++ views are valid only while
their owners remain alive and their buffers are not resized.
Python `*_into` processing functions require contiguous float64 NumPy arrays,
reject implicit conversion, and write into caller-owned output. Other Python
vector conversions are convenience operations and may copy.

Pileup is the documented pair-sum model in `processing/pileup.h`. It requires
positive acquisition times with live time <= real time and occupancy
`tau*total/real_time < 1`. Output includes full `2*N-1` support. Removal requires
that full support and verifies reconstructed counts. Truncated or noisy data
may have no supported inverse and are rejected. This model sums channel indices;
its applicability to an instrument must be validated separately.

Generation reuses a working experiment; it never normalizes concentrations
implicitly. Supply coupled composition rows explicitly. A complete dataset is
published only after all requested rows are accepted, recorded as failures, or
discarded. Cancellation and Stop failures leave an incomplete directory. A new
run rejects existing output directories. Caller-supplied rows are currently
resident; generated spectra and writer memory are bounded by one shard.



## Using the library

The [C++ generation example](../cpp/examples/generate.cpp) shows constructing
an experiment, declaring parameter columns and writing a dataset. Build it
with IBEAMLAB_BUILD_EXAMPLES=ON from the root.

See [model-format.md](model-format.md) for the package contract and the public
[metadata header](../cpp/include/ibeamlab/model/metadata.h) for directional schemas
and processing descriptors.
