#include "session.h"
#ifdef IBEAMLAB_HAS_ONNX
#include <array>
#include <onnxruntime_cxx_api.h>
namespace ibeamlab::inference::detail {
struct Session::Impl {
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::size_t in, out;
    std::string input_name, output_name;
    std::int64_t fixed_batch = -1;
    Impl(const model::Package &p, int threads)
        : in(p.metadata.input_width()), out(p.metadata.output_width()),
          input_name(p.metadata.input_name), output_name(p.metadata.output_name) {
        static Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "ibeamlab-cpp");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(threads);
        options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = Ort::Session(environment, p.graph.data(), p.graph.size(), options);
        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
            throw std::invalid_argument("graph must have one flat input and one flat output");
        Ort::AllocatorWithDefaultOptions allocator;
        if (input_name != session.GetInputNameAllocated(0, allocator).get() ||
            output_name != session.GetOutputNameAllocated(0, allocator).get())
            throw std::invalid_argument("graph tensor names differ from schema");
        auto check = [&](const Ort::TypeInfo &type, std::size_t width) {
            auto tensor = type.GetTensorTypeAndShapeInfo();
            auto shape = tensor.GetShape();
            if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                shape.size() != 2 ||
                (shape[1] >= 0 && shape[1] != static_cast<std::int64_t>(width)))
                throw std::invalid_argument("graph tensor shape/type mismatch");
            return shape[0];
        };
        fixed_batch = check(session.GetInputTypeInfo(0), in);
        auto output_batch = check(session.GetOutputTypeInfo(0), out);
        if (fixed_batch >= 0 && output_batch >= 0 && fixed_batch != output_batch)
            throw std::invalid_argument("graph batch dimensions mismatch");
        if (fixed_batch < 0)
            fixed_batch = output_batch;
    }
};
Session::Session(const model::Package &p, int threads) {
    if (threads < 1)
        throw std::invalid_argument("threads must be positive");
    p.validate();
    impl_ = std::make_unique<Impl>(p, threads);
}
Session::~Session() = default;
void Session::run(std::span<float> input, std::span<float> output, std::size_t batch) {
    auto &s = *impl_;
    if (!batch || input.size() / batch != s.in || input.size() % batch ||
        output.size() / batch != s.out || output.size() % batch ||
        (s.fixed_batch >= 0 && batch != static_cast<std::size_t>(s.fixed_batch)))
        throw std::invalid_argument("runtime batch shape mismatch");
    std::array<std::int64_t, 2> in_shape{static_cast<std::int64_t>(batch),
                                         static_cast<std::int64_t>(s.in)},
        out_shape{static_cast<std::int64_t>(batch), static_cast<std::int64_t>(s.out)};
    auto x =
        Ort::Value::CreateTensor<float>(s.memory, input.data(), input.size(), in_shape.data(), 2);
    auto y = Ort::Value::CreateTensor<float>(s.memory, output.data(), output.size(),
                                             out_shape.data(), 2);
    const char *in_name = s.input_name.c_str();
    const char *out_name = s.output_name.c_str();
    s.session.Run(Ort::RunOptions{nullptr}, &in_name, &x, 1, &out_name, &y, 1);
    for (float v : output)
        if (!std::isfinite(v))
            throw std::runtime_error("model returned nonfinite output");
}
} // namespace ibeamlab::inference::detail
#else
namespace ibeamlab::inference::detail {
struct Session::Impl {};
Session::Session(const model::Package &, int) {
    throw std::runtime_error("ONNX inference not enabled in this build");
}
Session::~Session() = default;
void Session::run(std::span<float>, std::span<float>, std::size_t) {
    throw std::runtime_error("ONNX inference unavailable");
}
} // namespace ibeamlab::inference::detail
#endif

namespace ibeamlab::inference::detail {
void run_graph(const model::Package &p, std::span<float> input, std::span<float> output,
               std::size_t batch, int threads) {
    Session session(p, threads);
    session.run(input, output, batch);
}
} // namespace ibeamlab::inference::detail
