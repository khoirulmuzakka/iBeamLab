#include <algorithm>
#include <ibeamlab/inference.h>
#include <ibeamlab/transforms.h>
#include <stdexcept>
#include <unordered_map>
#ifdef IBEAMLAB_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace ibeamlab::inference {
namespace {
#ifdef IBEAMLAB_HAS_ONNX
Ort::Env &onnxEnvironment() {
    static Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "ibeamlab"};
    return environment;
}
#endif
std::shared_ptr<const preprocessing::Transform> makeTransform(const model::TransformSpec &spec) {
    if (spec.type == "identity" || spec.type.empty())
        return std::make_shared<preprocessing::IdentityTransform>(spec.inputDimension);
    if (spec.type == "constant_factor")
        return std::make_shared<preprocessing::ConstantFactorTransform>(spec.inputDimension,
                                                                        spec.factor);
    if (spec.type == "standard_scaler")
        return std::make_shared<preprocessing::StandardScaler>(spec.mean, spec.deviation);
    if (spec.type == "min_max_scaler")
        return std::make_shared<preprocessing::MinMaxScaler>(spec.minimum, spec.scale, spec.low,
                                                             spec.high);
    if (spec.type == "log")
        return std::make_shared<preprocessing::LogTransform>(spec.inputDimension, spec.offset);
    if (spec.type == "pipeline") {
        auto pipeline = std::make_shared<preprocessing::TransformPipeline>();
        for (const auto &child : spec.transforms)
            pipeline->add(makeTransform(child));
        return pipeline;
    }
    throw std::runtime_error("unsupported transform type: " + spec.type);
}
preprocessing::Matrix flatten(const model::ModelMetadata &metadata,
                              const std::vector<std::vector<simulator::Spectrum>> &batch) {
    preprocessing::Matrix matrix;
    matrix.reserve(batch.size());
    for (const auto &sample : batch) {
        std::unordered_map<std::string, const simulator::Spectrum *> byLabel;
        for (const auto &spectrum : sample) {
            if (!byLabel.emplace(spectrum.label, &spectrum).second)
                throw std::invalid_argument("duplicate inference spectrum: " + spectrum.label);
        }
        std::vector<float> row;
        for (std::size_t i = 0; i < metadata.methodNames.size(); ++i) {
            const auto &method = metadata.methodNames[i];
            auto it = byLabel.find(method);
            if (it == byLabel.end())
                throw std::invalid_argument("missing inference spectrum: " + method);
            const auto expected = metadata.spectrumLengths.empty() ? it->second->counts.size()
                                                                   : metadata.spectrumLengths[i];
            if (it->second->counts.size() != expected)
                throw std::invalid_argument("spectrum length mismatch for method: " + method);
            row.insert(row.end(), it->second->counts.begin(), it->second->counts.end());
        }
        if (row.size() != metadata.inputDimension)
            throw std::invalid_argument("inference input dimension mismatch");
        matrix.push_back(std::move(row));
    }
    return matrix;
}
} // namespace
struct OnnxInferenceEngine::Impl {
    model::ModelPackage package;
    std::shared_ptr<const preprocessing::Transform> input, output;
#ifdef IBEAMLAB_HAS_ONNX
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
#endif
    Impl(model::ModelPackage value, InferenceOptions tuning)
        : package(std::move(value)), input(makeTransform(package.metadata().inputTransform)),
          output(makeTransform(package.metadata().outputTransform)) {
#ifdef IBEAMLAB_HAS_ONNX
        if (tuning.executionProvider != "cpu")
            throw std::invalid_argument("unsupported ONNX execution provider: " +
                                        tuning.executionProvider);
        if (tuning.intraOpThreads > 0)
            options.SetIntraOpNumThreads(tuning.intraOpThreads);
        if (tuning.interOpThreads > 0)
            options.SetInterOpNumThreads(tuning.interOpThreads);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.DisableMemPattern();
        options.DisableCpuMemArena();
        options.SetGraphOptimizationLevel(tuning.enableGraphOptimizations
                                              ? GraphOptimizationLevel::ORT_ENABLE_ALL
                                              : GraphOptimizationLevel::ORT_DISABLE_ALL);
        const auto &bytes = package.modelBytes();
        session = std::make_unique<Ort::Session>(onnxEnvironment(), bytes.data(), bytes.size(), options);
        if (session->GetInputCount() != 1 || session->GetOutputCount() != 1)
            throw std::runtime_error("ONNX model must have exactly one input and one output");
        Ort::AllocatorWithDefaultOptions allocator;
        const auto inputName = session->GetInputNameAllocated(0, allocator);
        const auto outputName = session->GetOutputNameAllocated(0, allocator);
        if (package.metadata().inputName != inputName.get() ||
            package.metadata().outputName != outputName.get())
            throw std::runtime_error("ONNX tensor names do not match package.toml");
        // ConstTensorTypeAndShapeInfo borrows from TypeInfo, so keep both owners alive.
        const auto inputType = session->GetInputTypeInfo(0);
        const auto outputType = session->GetOutputTypeInfo(0);
        const auto inputInfo = inputType.GetTensorTypeAndShapeInfo();
        const auto outputInfo = outputType.GetTensorTypeAndShapeInfo();
        if (inputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            outputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("ONNX input and output tensors must be float32");
        const auto inputShape = inputInfo.GetShape();
        const auto outputShape = outputInfo.GetShape();
        if (inputShape.size() != 2 || outputShape.size() != 2 ||
            inputShape[1] != static_cast<std::int64_t>(package.metadata().inputDimension) ||
            outputShape[1] != static_cast<std::int64_t>(package.metadata().outputDimension))
            throw std::runtime_error("ONNX tensor shapes do not match package.toml (input rank=" +
                std::to_string(inputShape.size()) + ", output rank=" +
                std::to_string(outputShape.size()) + ", input width=" +
                (inputShape.size()>1?std::to_string(inputShape[1]):"n/a") + ", output width=" +
                (outputShape.size()>1?std::to_string(outputShape[1]):"n/a") + ", expected=" +
                std::to_string(package.metadata().inputDimension) + "/" +
                std::to_string(package.metadata().outputDimension) + ")");
#else
        (void)tuning;
        throw std::runtime_error("iBeamLab was built without ONNX Runtime");
#endif
    }
};
OnnxInferenceEngine::OnnxInferenceEngine(model::ModelPackage package, InferenceOptions options)
    : impl_(std::make_unique<Impl>(std::move(package), options)) {}
OnnxInferenceEngine::~OnnxInferenceEngine() = default;
OnnxInferenceEngine::OnnxInferenceEngine(OnnxInferenceEngine &&) noexcept = default;
OnnxInferenceEngine &OnnxInferenceEngine::operator=(OnnxInferenceEngine &&) noexcept = default;
std::vector<InferenceResult>
OnnxInferenceEngine::predict(const std::vector<std::vector<simulator::Spectrum>> &batch) const {
#ifdef IBEAMLAB_HAS_ONNX
    auto matrix = impl_->input->apply(flatten(impl_->package.metadata(), batch));
    if (matrix.empty())
        return {};
    const auto width = impl_->package.metadata().inputDimension;
    std::vector<float> contiguous;
    contiguous.reserve(matrix.size() * width);
    for (const auto &row : matrix)
        contiguous.insert(contiguous.end(), row.begin(), row.end());
    std::array<std::int64_t, 2> shape{static_cast<std::int64_t>(matrix.size()),
                                      static_cast<std::int64_t>(width)};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(memory, contiguous.data(), contiguous.size(),
                                                  shape.data(), shape.size());
    const char *inputs[]{impl_->package.metadata().inputName.c_str()};
    const char *outputs[]{impl_->package.metadata().outputName.c_str()};
    auto values = impl_->session->Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
    if (values.size() != 1 || !values[0].IsTensor())
        throw std::runtime_error("ONNX model did not return one tensor");
    const auto info = values[0].GetTensorTypeAndShapeInfo();
    const auto outputShape = info.GetShape();
    if (outputShape.size() != 2 || static_cast<std::size_t>(outputShape[0]) != matrix.size())
        throw std::runtime_error("ONNX output batch shape mismatch");
    const auto widthOut = static_cast<std::size_t>(outputShape[1]);
    if (widthOut != impl_->package.metadata().outputDimension)
        throw std::runtime_error("ONNX output dimension mismatch");
    const float *data = values[0].GetTensorData<float>();
    preprocessing::Matrix transformed(matrix.size(), std::vector<float>(widthOut));
    for (std::size_t r = 0; r < matrix.size(); ++r)
        std::copy_n(data + r * widthOut, widthOut, transformed[r].begin());
    auto physical = impl_->output->inverse(transformed);
    std::vector<InferenceResult> results(matrix.size());
    for (std::size_t r = 0; r < results.size(); ++r)
        for (std::size_t c = 0; c < widthOut; ++c)
            results[r].values.push_back({
                impl_->package.metadata().outputFeatures.at(c), physical[r][c],
                impl_->package.metadata().outputUnits.empty()
                    ? std::string{}
                    : impl_->package.metadata().outputUnits.at(c)});
    return results;
#else
    (void)batch;
    throw std::runtime_error("iBeamLab was built without ONNX Runtime");
#endif
}
} // namespace ibeamlab::inference
