#pragma once
#include <ibeamlab/inference.h>
#include <ibeamlab/model_package.h>
#include <ibeamlab/transforms.h>
#include <memory>
namespace ibeamlab::inference::detail {
struct OnnxPrediction {
    preprocessing::Matrix values, presenceProbability, posteriorStd;
};
class OnnxModel {
  public:
    OnnxModel(model::ModelPackage package, InferenceOptions options);
    ~OnnxModel();
    const model::ModelMetadata &metadata() const noexcept;
    preprocessing::Matrix run(const preprocessing::Matrix &physicalInput) const;
    OnnxPrediction runPrediction(const preprocessing::Matrix &physicalInput) const;
    preprocessing::Matrix runSequence(const std::vector<float> &physicalAd,
                                      std::size_t batch, std::size_t layers) const;

  private:
    OnnxPrediction runTensor(const std::vector<float> &values,
                             const std::vector<std::int64_t> &shape) const;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ibeamlab::inference::detail
