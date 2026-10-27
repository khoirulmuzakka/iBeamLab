#include "onnx_model.h"
#include <algorithm>
#include <array>
#include <stdexcept>
#ifdef IBEAMLAB_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif
namespace ibeamlab::inference::detail {
namespace {
#ifdef IBEAMLAB_HAS_ONNX
Ort::Env &environment() {
    static Ort::Env value{ORT_LOGGING_LEVEL_WARNING, "ibeamlab"};
    return value;
}
#endif
std::shared_ptr<const preprocessing::Transform> makeTransform(const model::TransformSpec &s) {
    if (s.type.empty() || s.type == "identity")
        return std::make_shared<preprocessing::IdentityTransform>(s.inputDimension);
    if (s.type == "constant_factor")
        return std::make_shared<preprocessing::ConstantFactorTransform>(s.inputDimension, s.factor);
    if (s.type == "standard_scaler")
        return std::make_shared<preprocessing::StandardScaler>(s.mean, s.deviation);
    if (s.type == "min_max_scaler")
        return std::make_shared<preprocessing::MinMaxScaler>(s.minimum, s.scale, s.low, s.high);
    if (s.type == "parameter_bound_min_max_scaler")
        return std::make_shared<preprocessing::ParameterBoundMinMaxScaler>(s.minimum, s.scale,
                                                                           s.low, s.high);
    if (s.type == "layerwise_concentration_normalizer")
        return std::make_shared<preprocessing::LayerwiseConcentrationNormalizer>(
            s.inputDimension, s.concentrationGroups);
    if (s.type == "log")
        return std::make_shared<preprocessing::LogTransform>(s.inputDimension, s.offset);
    if (s.type == "pipeline") {
        auto p = std::make_shared<preprocessing::TransformPipeline>();
        for (const auto &c : s.transforms)
            p->add(makeTransform(c));
        return p;
    }
    throw std::runtime_error("unsupported transform type: " + s.type);
}
} // namespace
struct OnnxModel::Impl {
    model::ModelPackage package;
    std::vector<std::string> outputNames;
    std::shared_ptr<const preprocessing::Transform> input, output;
#ifdef IBEAMLAB_HAS_ONNX
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
#endif
    Impl(model::ModelPackage p, InferenceOptions tuning)
        : package(std::move(p)),
          input(package.metadata().forward.layerInput.dynamicLayers ? nullptr :
                    makeTransform(package.metadata().inputTransform)),
          output(makeTransform(package.metadata().outputTransform)) {
#ifdef IBEAMLAB_HAS_ONNX
        if (tuning.executionProvider != "cpu" && tuning.executionProvider != "cuda")
            throw std::invalid_argument("unsupported ONNX execution provider");
        if (tuning.intraOpThreads > 0)
            options.SetIntraOpNumThreads(tuning.intraOpThreads);
        if (tuning.interOpThreads > 0)
            options.SetInterOpNumThreads(tuning.interOpThreads);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(tuning.enableGraphOptimizations
                                              ? GraphOptimizationLevel::ORT_ENABLE_ALL
                                              : GraphOptimizationLevel::ORT_DISABLE_ALL);
        if (tuning.executionProvider == "cuda") {
            if (tuning.deviceId < 0)
                throw std::invalid_argument("CUDA deviceId must be nonnegative");
#ifdef IBEAMLAB_HAS_CUDA
            try {
                OrtCUDAProviderOptions cuda{};
                cuda.device_id = tuning.deviceId;
                options.AppendExecutionProvider_CUDA(cuda);
            } catch (const Ort::Exception &e) {
                throw std::runtime_error(std::string("Cannot initialize ONNX CUDA provider for device ") +
                    std::to_string(tuning.deviceId) +
                    ". Check the NVIDIA driver, GPU ONNX Runtime, CUDA and cuDNN DLLs: " + e.what());
            }
#else
            throw std::runtime_error("CUDA requested but iBeamLab was built for CPU ONNX Runtime. "
                                     "Configure IBEAMLAB_ONNX_PROVIDER=CUDA.");
#endif
        }
        const auto &b = package.modelBytes();
        try {
            session = std::make_unique<Ort::Session>(environment(), b.data(), b.size(), options);
        } catch (const Ort::Exception &e) {
            throw std::runtime_error("Cannot create ONNX " + tuning.executionProvider +
                                     " session: " + e.what());
        }
        const auto &m = package.metadata();
        outputNames = {m.outputName};
        if (m.modelType == model::ModelType::Inverse && m.formatVersion == 4) {
            outputNames.push_back(m.inverse.presenceProbabilityOutputName);
            outputNames.push_back(m.inverse.posteriorStdOutputName);
        }
        if (session->GetInputCount() != 1 || session->GetOutputCount() != outputNames.size())
            throw std::runtime_error("ONNX input/output count differs from metadata");
        Ort::AllocatorWithDefaultOptions a;
        auto in = session->GetInputNameAllocated(0, a);
        if (m.inputName != in.get())
            throw std::runtime_error("ONNX input name differs from metadata");
        const auto inputType = session->GetInputTypeInfo(0);
        const auto is = inputType.GetTensorTypeAndShapeInfo();
        const auto ish = is.GetShape();
        const bool sequence = m.modelType == model::ModelType::Forward && m.forward.layerInput.dynamicLayers;
        const bool shapeMatches = sequence
            ? (ish.size() == 3 && ish[0] == -1 && ish[1] == -1 &&
               ish[2] == static_cast<std::int64_t>(m.forward.layerInput.elements.size()))
            : (ish.size() == 2 && ish[1] == static_cast<std::int64_t>(m.inputDimension));
        if (is.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || !shapeMatches)
            throw std::runtime_error("ONNX input schema differs from metadata");
        for (const auto &name : outputNames) {
            bool found = false;
            for (std::size_t i = 0; i < session->GetOutputCount(); ++i) {
                auto actual = session->GetOutputNameAllocated(i, a);
                if (name != actual.get()) continue;
                found = true;
                const auto type = session->GetOutputTypeInfo(i);
                const auto info = type.GetTensorTypeAndShapeInfo();
                const auto shape = info.GetShape();
                if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                    shape.size() != 2 || shape[1] != static_cast<std::int64_t>(m.outputDimension))
                    throw std::runtime_error("ONNX output schema differs from metadata");
            }
            if (!found) throw std::runtime_error("ONNX output name differs from metadata");
        }
#else
        (void)tuning;
        throw std::runtime_error("iBeamLab was built without ONNX Runtime");
#endif
    }
};
OnnxModel::OnnxModel(model::ModelPackage p, InferenceOptions o)
    : impl_(std::make_unique<Impl>(std::move(p), o)) {}
OnnxModel::~OnnxModel() = default;
const model::ModelMetadata &OnnxModel::metadata() const noexcept {
    return impl_->package.metadata();
}
preprocessing::Matrix OnnxModel::run(const preprocessing::Matrix &physical) const {
    return runPrediction(physical).values;
}
OnnxPrediction OnnxModel::runPrediction(const preprocessing::Matrix &physical) const {
    if (!impl_->input)
        throw std::invalid_argument("dynamic AD models require rank-three sequence input");
    auto matrix = impl_->input->apply(physical);
    if (matrix.empty())
        return {};
    const auto width = impl_->package.metadata().inputDimension;
    std::vector<float> data;
    data.reserve(matrix.size() * width);
    for (const auto &r : matrix)
        data.insert(data.end(), r.begin(), r.end());
    return runTensor(data, {static_cast<std::int64_t>(matrix.size()),
                            static_cast<std::int64_t>(width)});
}
preprocessing::Matrix OnnxModel::runSequence(const std::vector<float> &physicalAd,
                                            std::size_t batch, std::size_t layers) const {
    if (!impl_->package.metadata().forward.layerInput.dynamicLayers)
        throw std::invalid_argument("model does not support dynamic layer input");
    if (!batch) return {};
    if (!layers) throw std::invalid_argument("layer sequence cannot be empty");
    return runTensor(physicalAd, {static_cast<std::int64_t>(batch),
                                  static_cast<std::int64_t>(layers),
                                  static_cast<std::int64_t>(impl_->package.metadata().forward.layerInput.elements.size())}).values;
}
OnnxPrediction OnnxModel::runTensor(const std::vector<float> &data,
                                   const std::vector<std::int64_t> &shape) const {
#ifdef IBEAMLAB_HAS_ONNX
    const auto batch = static_cast<std::size_t>(shape.at(0));
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(memory, const_cast<float *>(data.data()), data.size(), shape.data(),
                                                  shape.size());
    const char *ins[]{impl_->package.metadata().inputName.c_str()};
    std::vector<const char *> outs;
    for (const auto &name : impl_->outputNames) outs.push_back(name.c_str());
    auto values = impl_->session->Run(Ort::RunOptions{nullptr}, ins, &tensor, 1,
                                     outs.data(), outs.size());
    const auto outputWidth = impl_->package.metadata().outputDimension;
    auto extract = [&](std::size_t index) {
        const auto info = values.at(index).GetTensorTypeAndShapeInfo();
        const auto outputShape = info.GetShape();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            outputShape.size() != 2 || outputShape[0] != static_cast<std::int64_t>(batch) ||
            outputShape[1] != static_cast<std::int64_t>(outputWidth))
            throw std::runtime_error("ONNX output shape mismatch");
        const float *raw = values[index].GetTensorData<float>();
        preprocessing::Matrix result(batch, std::vector<float>(outputWidth));
        for (std::size_t r = 0; r < batch; ++r)
            std::copy_n(raw + r * outputWidth, outputWidth, result[r].begin());
        return result;
    };
    OnnxPrediction result;
    result.values = impl_->output->inverse(extract(0));
    if (values.size() == 3) {
        result.presenceProbability = extract(1);
        result.posteriorStd = extract(2);
        for (auto &row : result.posteriorStd)
            for (auto &value : row)
                value = static_cast<float>(value / impl_->package.metadata().inverse.posteriorStdInverseFactor);
    }
    return result;
#else
    (void)data;
    (void)shape;
    throw std::runtime_error("iBeamLab was built without ONNX Runtime");
#endif
}
} // namespace ibeamlab::inference::detail
