#include "onnx_model.h"
#include <algorithm>
#include <array>
#include <stdexcept>
#ifdef IBEAMLAB_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif
namespace ibeamlab::inference::detail { namespace {
#ifdef IBEAMLAB_HAS_ONNX
Ort::Env& environment(){static Ort::Env value{ORT_LOGGING_LEVEL_WARNING,"ibeamlab"};return value;}
#endif
std::shared_ptr<const preprocessing::Transform> makeTransform(const model::TransformSpec&s){
    if(s.type.empty()||s.type=="identity")return std::make_shared<preprocessing::IdentityTransform>(s.inputDimension);
    if(s.type=="constant_factor")return std::make_shared<preprocessing::ConstantFactorTransform>(s.inputDimension,s.factor);
    if(s.type=="standard_scaler")return std::make_shared<preprocessing::StandardScaler>(s.mean,s.deviation);
    if(s.type=="min_max_scaler")return std::make_shared<preprocessing::MinMaxScaler>(s.minimum,s.scale,s.low,s.high);
    if(s.type=="log")return std::make_shared<preprocessing::LogTransform>(s.inputDimension,s.offset);
    if(s.type=="pipeline"){auto p=std::make_shared<preprocessing::TransformPipeline>();for(const auto&c:s.transforms)p->add(makeTransform(c));return p;}
    throw std::runtime_error("unsupported transform type: "+s.type);
}
}
struct OnnxModel::Impl{
    model::ModelPackage package;std::shared_ptr<const preprocessing::Transform>input,output;
#ifdef IBEAMLAB_HAS_ONNX
    Ort::SessionOptions options;std::unique_ptr<Ort::Session>session;
#endif
    Impl(model::ModelPackage p,InferenceOptions tuning):package(std::move(p)),input(makeTransform(package.metadata().inputTransform)),output(makeTransform(package.metadata().outputTransform)){
#ifdef IBEAMLAB_HAS_ONNX
        if(tuning.executionProvider!="cpu")throw std::invalid_argument("unsupported ONNX execution provider");
        if(tuning.intraOpThreads>0)options.SetIntraOpNumThreads(tuning.intraOpThreads);if(tuning.interOpThreads>0)options.SetInterOpNumThreads(tuning.interOpThreads);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);options.SetGraphOptimizationLevel(tuning.enableGraphOptimizations?GraphOptimizationLevel::ORT_ENABLE_ALL:GraphOptimizationLevel::ORT_DISABLE_ALL);
        const auto&b=package.modelBytes();session=std::make_unique<Ort::Session>(environment(),b.data(),b.size(),options);
        if(session->GetInputCount()!=1||session->GetOutputCount()!=1)throw std::runtime_error("ONNX model must have one input and output");
        Ort::AllocatorWithDefaultOptions a;auto in=session->GetInputNameAllocated(0,a);auto out=session->GetOutputNameAllocated(0,a);if(package.metadata().inputName!=in.get()||package.metadata().outputName!=out.get())throw std::runtime_error("ONNX tensor names differ from metadata");
        const auto inputType=session->GetInputTypeInfo(0);const auto outputType=session->GetOutputTypeInfo(0);const auto is=inputType.GetTensorTypeAndShapeInfo();const auto os=outputType.GetTensorTypeAndShapeInfo();const auto ish=is.GetShape(),osh=os.GetShape();if(is.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||os.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||ish.size()!=2||osh.size()!=2||ish[1]!=static_cast<std::int64_t>(package.metadata().inputDimension)||osh[1]!=static_cast<std::int64_t>(package.metadata().outputDimension))throw std::runtime_error("ONNX tensor schema differs from metadata");
#else
        (void)tuning;throw std::runtime_error("iBeamLab was built without ONNX Runtime");
#endif
    }
};
OnnxModel::OnnxModel(model::ModelPackage p,InferenceOptions o):impl_(std::make_unique<Impl>(std::move(p),o)){}OnnxModel::~OnnxModel()=default;
const model::ModelMetadata&OnnxModel::metadata()const noexcept{return impl_->package.metadata();}
preprocessing::Matrix OnnxModel::run(const preprocessing::Matrix&physical)const{
#ifdef IBEAMLAB_HAS_ONNX
    auto matrix=impl_->input->apply(physical);if(matrix.empty())return{};const auto width=impl_->package.metadata().inputDimension;std::vector<float>data;data.reserve(matrix.size()*width);for(const auto&r:matrix)data.insert(data.end(),r.begin(),r.end());std::array<std::int64_t,2>shape{static_cast<std::int64_t>(matrix.size()),static_cast<std::int64_t>(width)};auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);auto tensor=Ort::Value::CreateTensor<float>(memory,data.data(),data.size(),shape.data(),shape.size());const char*ins[]{impl_->package.metadata().inputName.c_str()};const char*outs[]{impl_->package.metadata().outputName.c_str()};auto values=impl_->session->Run(Ort::RunOptions{nullptr},ins,&tensor,1,outs,1);const auto info=values.at(0).GetTensorTypeAndShapeInfo();const auto outputShape=info.GetShape();if(outputShape.size()!=2||static_cast<std::size_t>(outputShape[0])!=matrix.size()||static_cast<std::size_t>(outputShape[1])!=impl_->package.metadata().outputDimension)throw std::runtime_error("ONNX output shape mismatch");const auto outputWidth=impl_->package.metadata().outputDimension;const float*raw=values[0].GetTensorData<float>();preprocessing::Matrix transformed(matrix.size(),std::vector<float>(outputWidth));for(std::size_t r=0;r<matrix.size();++r)std::copy_n(raw+r*outputWidth,outputWidth,transformed[r].begin());return impl_->output->inverse(transformed);
#else
    (void)physical;throw std::runtime_error("iBeamLab was built without ONNX Runtime");
#endif
}
}
