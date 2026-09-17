#include <ibeamlab/forward_model.h>
#include <ibeamlab/parameter.h>
#include "onnx_model.h"
#include <cmath>
#include <stdexcept>
namespace ibeamlab::inference {
struct ForwardModel::Impl{detail::OnnxModel model;explicit Impl(model::ModelPackage p,InferenceOptions o):model(std::move(p),o){if(model.metadata().modelType!=ibeamlab::model::ModelType::Forward)throw std::invalid_argument("ForwardModel requires a forward package");}};
ForwardModel::ForwardModel(const std::filesystem::path&p,InferenceOptions o):ForwardModel(model::ModelPackage::open(p),o){}ForwardModel::ForwardModel(model::ModelPackage p,InferenceOptions o):impl_(std::make_unique<Impl>(std::move(p),o)){}ForwardModel::~ForwardModel()=default;ForwardModel::ForwardModel(ForwardModel&&)noexcept=default;ForwardModel&ForwardModel::operator=(ForwardModel&&)noexcept=default;
std::vector<ForwardResult>ForwardModel::predict(const std::vector<simulator::SimulationInput>&inputs)const{preprocessing::Matrix rows;rows.reserve(inputs.size());const auto&m=impl_->model.metadata().forward;for(const auto&i:inputs){i.sample.validate();i.setup.validate();std::vector<float>row;row.reserve(m.inputParameters.size());for(const auto&p:m.inputParameters){const auto v=parameter::read(i,p.target);if(!std::isfinite(v)||v<p.lowerBound||v>p.upperBound)throw std::invalid_argument("forward parameter outside bounds: "+p.name);row.push_back(static_cast<float>(v));}rows.push_back(std::move(row));}const auto values=impl_->model.run(rows);std::vector<ForwardResult>result(values.size());for(std::size_t r=0;r<values.size();++r){std::size_t offset=0;for(const auto&s:m.outputSpectra){simulator::Spectrum spectrum{s.label,{}};spectrum.counts.assign(values[r].begin()+static_cast<std::ptrdiff_t>(offset),values[r].begin()+static_cast<std::ptrdiff_t>(offset+s.length));result[r].spectra.push_back(std::move(spectrum));offset+=s.length;}}return result;}
}
