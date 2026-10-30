#pragma once
#include <ibeamlab/export.h>
#include <string>
#include <vector>

namespace ibeamlab::inference {
/** @brief Requested ONNX execution backend; CUDA also permits CPU control/shape operators. */
enum class ExecutionProvider { Cpu, Cuda };
/** @brief Per-session execution tuning, independent of scientific package metadata.
 * A CUDA-enabled build can create both CPU and CUDA sessions. Requesting CUDA
 * never silently selects a CPU-only session if CUDA initialization fails.
 */
struct IBEAMLAB_API InferenceOptions {
    ExecutionProvider execution_provider = ExecutionProvider::Cpu;
    int device_id = 0; ///< CUDA device ordinal; ignored for CPU sessions.
    int intra_op_threads = 1;
    int inter_op_threads = 1;
    bool enable_graph_optimizations = true;
    bool allow_tf32 = false; ///< CUDA only: opt in to reduced-mantissa tensor-core arithmetic.
    bool apply_pileup = true; ///< Forward only: apply declared Pileup steps; false skips them.
    /** @brief Reject unknown providers, negative devices and nonpositive thread counts. */
    void validate() const;
};
/** @brief Provider names usable by this build/runtime, without creating a session.
 * Returns "cpu" and optionally "cuda"; an empty result means ONNX is disabled.
 * A listed provider still requires working device drivers and dependent libraries.
 */
IBEAMLAB_API std::vector<std::string> available_execution_providers();
} // namespace ibeamlab::inference
