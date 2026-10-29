#include "session.h"
#ifdef IBEAMLAB_HAS_ONNX
#include <array>
#include <limits>
#include <onnxruntime_cxx_api.h>
namespace ibeamlab::inference::detail {
struct Session::Impl {
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::size_t in, out;
    std::string input_name, output_name;
    std::int64_t fixed_batch = -1;
    bool layer_input;
    Ort::Value input_tensor{nullptr}, output_tensor{nullptr};
    const float *input_pointer = nullptr;
    float *output_pointer = nullptr;
    std::size_t input_batch = 0, input_layers = 0, output_batch = 0;
    std::string provider_name;
    Impl(const model::Package &p, const InferenceOptions &tuning)
        : in(p.metadata.input_width()), out(p.metadata.output_width()),
          input_name(p.metadata.input_name), output_name(p.metadata.output_name),
          layer_input(p.metadata.layer_input()),
          provider_name(tuning.execution_provider == ExecutionProvider::Cuda ? "CUDA" : "CPU") {
        static Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "ibeamlab-cpp");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(tuning.intra_op_threads);
        options.SetInterOpNumThreads(tuning.inter_op_threads);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(tuning.enable_graph_optimizations
            ? GraphOptimizationLevel::ORT_ENABLE_ALL : GraphOptimizationLevel::ORT_DISABLE_ALL);
        if (tuning.execution_provider == ExecutionProvider::Cuda) {
#ifdef IBEAMLAB_HAS_CUDA
            try {
                const auto &api = Ort::GetApi();
                OrtCUDAProviderOptionsV2 *raw = nullptr;
                Ort::ThrowOnError(api.CreateCUDAProviderOptions(&raw));
                std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)>
                    cuda(raw, api.ReleaseCUDAProviderOptions);
                const auto device = std::to_string(tuning.device_id);
                const char *keys[]{"device_id", "do_copy_in_default_stream", "use_tf32"};
                const char *values[]{device.c_str(), "1", tuning.allow_tf32 ? "1" : "0"};
                Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda.get(), keys, values, 3));
                options.AppendExecutionProvider_CUDA_V2(*cuda);
            } catch (const Ort::Exception &error) {
                throw std::runtime_error("Cannot initialize ONNX CUDA device " + std::to_string(tuning.device_id) +
                    "; check GPU ONNX Runtime, NVIDIA drivers, CUDA and cuDNN libraries: " + error.what());
            }
#else
            throw std::runtime_error("CUDA requested but this build supports CPU ONNX Runtime only; "
                                     "configure IBEAMLAB_ONNX_PROVIDER=CUDA with a GPU runtime SDK");
#endif
        }
        try {
            session = Ort::Session(environment, p.graph.data(), p.graph.size(), options);
        } catch (const Ort::Exception &error) {
            throw std::runtime_error("Cannot create ONNX " + provider_name + " session for device " +
                std::to_string(tuning.device_id) + ": " + error.what());
        }
        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
            throw std::invalid_argument("graph must have one input and one packed spectrum/EDP output");
        Ort::AllocatorWithDefaultOptions allocator;
        if (input_name != session.GetInputNameAllocated(0, allocator).get() ||
            output_name != session.GetOutputNameAllocated(0, allocator).get())
            throw std::invalid_argument("graph tensor names differ from schema");
        auto check = [&](const Ort::TypeInfo &type, std::size_t width, bool sequence) {
            auto tensor = type.GetTensorTypeAndShapeInfo();
            auto shape = tensor.GetShape();
            if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                shape.size() != (sequence ? 3u : 2u) ||
                shape.back() != static_cast<std::int64_t>(width))
                throw std::invalid_argument("graph tensor shape/type mismatch");
            if (sequence && (shape[0] >= 0 || shape[1] >= 0))
                throw std::invalid_argument("layer graph must have dynamic batch and layer axes");
            return shape[0];
        };
        fixed_batch = check(session.GetInputTypeInfo(0), in, layer_input);
        auto declared_output_batch = check(session.GetOutputTypeInfo(0), out, false);
        if (layer_input && declared_output_batch >= 0)
            throw std::invalid_argument("layer graph output must have a dynamic batch axis");
        if (fixed_batch >= 0 && declared_output_batch >= 0 && fixed_batch != declared_output_batch)
            throw std::invalid_argument("graph batch dimensions mismatch");
        if (fixed_batch < 0)
            fixed_batch = declared_output_batch;
    }
};
Session::Session(const model::Package &p, InferenceOptions tuning) : options_(tuning) {
    tuning.validate();
    p.validate();
    impl_ = std::make_unique<Impl>(p, tuning);
}
Session::~Session() = default;
void Session::run(std::span<const float> input, std::span<float> output, std::size_t batch,
                  std::size_t layers) {
    auto &s = *impl_;
    if (!batch || !layers || (!s.layer_input && layers != 1) ||
        layers > std::numeric_limits<std::size_t>::max() / s.in ||
        input.size() / batch != s.in * layers || input.size() % batch ||
        output.size() / batch != s.out || output.size() % batch ||
        (s.fixed_batch >= 0 && batch != static_cast<std::size_t>(s.fixed_batch)))
        throw std::invalid_argument("runtime batch shape mismatch");
    std::array<std::int64_t, 3> in_shape{static_cast<std::int64_t>(batch),
        static_cast<std::int64_t>(s.layer_input ? layers : s.in), static_cast<std::int64_t>(s.in)};
    std::array<std::int64_t, 2> out_shape{static_cast<std::int64_t>(batch), static_cast<std::int64_t>(s.out)};
    // Tensor wrappers borrow caller memory. Retain their OrtValue handles while
    // addresses and shapes stay valid; no wrapper construction on a warm call.
    if (s.input_pointer != input.data() || s.input_batch != batch || s.input_layers != layers) {
        s.input_tensor = Ort::Value::CreateTensor<float>(s.memory, const_cast<float *>(input.data()),
            input.size(), in_shape.data(), s.layer_input ? 3 : 2);
        s.input_pointer = input.data();
        s.input_batch = batch;
        s.input_layers = layers;
    }
    if (s.output_pointer != output.data() || s.output_batch != batch) {
        s.output_tensor = Ort::Value::CreateTensor<float>(s.memory, output.data(), output.size(), out_shape.data(), 2);
        s.output_pointer = output.data();
        s.output_batch = batch;
    }
    const char *in_name = s.input_name.c_str();
    const char *out_name = s.output_name.c_str();
    // CPU tensors preserve the host API. For CUDA, ORT manages the transfers and
    // reuses its device arena; Run completes before host postprocessing begins.
    try {
        s.session.Run(Ort::RunOptions{nullptr}, &in_name, &s.input_tensor, 1, &out_name, &s.output_tensor, 1);
    } catch (const Ort::Exception &error) {
        throw std::runtime_error("ONNX " + s.provider_name + " inference failed: " + error.what());
    }
    for (float v : output)
        if (!std::isfinite(v))
            throw std::runtime_error("model returned nonfinite output");
}
} // namespace ibeamlab::inference::detail
#else
namespace ibeamlab::inference::detail {
struct Session::Impl {};
Session::Session(const model::Package &, InferenceOptions tuning) : options_(tuning) {
    tuning.validate();
    throw std::runtime_error("ONNX inference not enabled in this build");
}
Session::~Session() = default;
void Session::run(std::span<const float>, std::span<float>, std::size_t, std::size_t) {
    throw std::runtime_error("ONNX inference unavailable");
}
} // namespace ibeamlab::inference::detail
#endif

namespace ibeamlab::inference::detail {
const InferenceOptions &Session::options() const { return options_; }
void run_graph(const model::Package &p, std::span<const float> input, std::span<float> output,
               std::size_t batch, InferenceOptions options, std::size_t layers) {
    Session session(p, options);
    session.run(input, output, batch, layers);
}
} // namespace ibeamlab::inference::detail
