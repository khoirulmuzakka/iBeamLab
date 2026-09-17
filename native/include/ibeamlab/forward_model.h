#pragma once
/** @file forward_model.h @brief Executes a trained sample/setup-to-spectra ONNX model. */
#include <ibeamlab/inference.h>
#include <ibeamlab/model_package.h>
#include <ibeamlab/simulator.h>
#include <filesystem>
#include <memory>
#include <vector>
namespace ibeamlab::inference {
struct ForwardResult { std::vector<simulator::Spectrum> spectra; };
/** @brief Deployment forward inference; independent of DataGenerator and ISimulator. */
class ForwardModel {
public:
    explicit ForwardModel(const std::filesystem::path& package, InferenceOptions options = {});
    explicit ForwardModel(model::ModelPackage package, InferenceOptions options = {});
    ~ForwardModel(); ForwardModel(ForwardModel&&) noexcept; ForwardModel& operator=(ForwardModel&&) noexcept;
    std::vector<ForwardResult> predict(const std::vector<simulator::SimulationInput>& inputs) const;
private: struct Impl; std::unique_ptr<Impl> impl_;
};
}
