#include <ibeamlab/inference/options.h>
#include <stdexcept>
#include <algorithm>
#ifdef IBEAMLAB_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace ibeamlab::inference {
void InferenceOptions::validate() const {
    if (execution_provider != ExecutionProvider::Cpu && execution_provider != ExecutionProvider::Cuda)
        throw std::invalid_argument("unsupported ONNX execution provider");
    if (device_id < 0) throw std::invalid_argument("device_id must be nonnegative");
    if (intra_op_threads < 1) throw std::invalid_argument("intra_op_threads must be positive");
    if (inter_op_threads < 1) throw std::invalid_argument("inter_op_threads must be positive");
}
std::vector<std::string> available_execution_providers() {
    std::vector<std::string> result;
#ifdef IBEAMLAB_HAS_ONNX
    const auto providers = Ort::GetAvailableProviders();
    if (std::find(providers.begin(), providers.end(), "CPUExecutionProvider") != providers.end())
        result.emplace_back("cpu");
#ifdef IBEAMLAB_HAS_CUDA
    if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end())
        result.emplace_back("cuda");
#endif
#endif
    return result;
}
} // namespace ibeamlab::inference
