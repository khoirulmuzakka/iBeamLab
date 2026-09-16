# Native third-party dependencies

- pybind11 — BSD-3-Clause, used only by `_ibeamlab_cpp`.
- toml++ — MIT, used to read and write language-neutral TOML metadata.
- miniz — MIT, used to read model ZIP files and validate CRC32 checksums.
- ONNX Runtime — MIT, optional inference runtime. The Windows fallback is pinned
  by `IBEAMLAB_ONNX_VERSION`.
- OpenMP — compiler runtime, optional and private to preprocessing.
- SIMNRA — external Windows application automated through COM; it is not bundled.

Dependency source distributions retain their upstream license files in CMake's
build dependency directory. Downstream binary distributors remain responsible for
shipping the notices required by the exact dependencies they include.
