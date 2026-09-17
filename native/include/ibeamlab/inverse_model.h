#pragma once
/** @file inverse_model.h @brief Executes a trained spectra-to-sample ONNX model. */
#include <ibeamlab/inference.h>
#include <ibeamlab/model_package.h>
#include <ibeamlab/simulator.h>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
namespace ibeamlab::inference {
struct NamedValue { std::string name; float value{}; std::string unit; };
struct InverseResult { sample::SampleModel sample; std::vector<NamedValue> parameters; };
class InverseModel {
public:
    explicit InverseModel(const std::filesystem::path& package, InferenceOptions options = {});
    explicit InverseModel(model::ModelPackage package, InferenceOptions options = {});
    ~InverseModel(); InverseModel(InverseModel&&) noexcept; InverseModel& operator=(InverseModel&&) noexcept;
    std::vector<InverseResult> predict(const std::vector<std::vector<simulator::Spectrum>>& batch) const;
private: struct Impl; std::unique_ptr<Impl> impl_;
};
}
