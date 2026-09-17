#pragma once
#include <ibeamlab/inference.h>
#include <ibeamlab/model_package.h>
#include <ibeamlab/transforms.h>
#include <memory>
namespace ibeamlab::inference::detail {
class OnnxModel {
public:
    OnnxModel(model::ModelPackage package, InferenceOptions options); ~OnnxModel();
    const model::ModelMetadata& metadata() const noexcept;
    preprocessing::Matrix run(const preprocessing::Matrix& physicalInput) const;
private: struct Impl; std::unique_ptr<Impl> impl_;
};
}
