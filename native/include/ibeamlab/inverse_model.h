#pragma once
/** @file inverse_model.h @brief Executes a trained spectra-to-EDP ONNX model. */
#include <filesystem>
#include <ibeamlab/export.h>
#include <ibeamlab/inference.h>
#include <ibeamlab/model_package.h>
#include <ibeamlab/simulator.h>
#include <memory>
#include <string>
#include <vector>
namespace ibeamlab::inference {
/** @brief Elemental areal densities; values[layer][element] includes trailing padding. */
struct IBEAMLAB_API EdpMap {
    using Matrix = std::vector<std::vector<float>>;
    std::vector<std::string> elements;
    std::string unit{"1e15 atoms/cm2"};
    Matrix values;
    Matrix presenceProbability; // Dimensionless [0, 1], same layout as values.
    Matrix posteriorStd; // Full posterior standard deviation, in unit.
    bool uncertaintyPredicted{false}; // False means a deterministic convention.
    /** @brief Validates all three mandatory matrices and their shared layout. */
    void validate() const;
    /** @brief Reconstructs thickness/composition, preserving template fixed properties.
     * Removes trailing zero layers; rejects an entirely empty profile.
     */
    sample::SampleModel toSample(const sample::SampleModel &sampleTemplate) const;
};
struct InverseResult {
    EdpMap edp;
};
/** @brief Spectra and the experimental settings under which they were measured. */
struct InverseInput {
    std::vector<simulator::Spectrum> spectra;
    sample::ExperimentalSetup setup;
    /** @brief All spectra are already pileup-free, with live-time scaling undone.
     * Also set true for simulated spectra generated without pileup.
     */
    bool pileupAlreadyRemoved{false};
};
class IBEAMLAB_API InverseModel {
  public:
    explicit InverseModel(const std::filesystem::path &package, InferenceOptions options = {});
    explicit InverseModel(model::ModelPackage package, InferenceOptions options = {});
    ~InverseModel();
    InverseModel(InverseModel &&) noexcept;
    InverseModel &operator=(InverseModel &&) noexcept;
    /** @brief Corrects experimental spectra and predicts physical EDPs in batch order.
     * Removes pileup when required, rebins to the training grid, normalizes
     * ParticlesSr, then applies packaged transforms and ONNX. Requires coverage
     * of the training energy grid. Beam/resolution mismatches warn and are not
     * corrected. Invalid input/output aborts the batch with an exception.
     */
    std::vector<InverseResult> predict(const std::vector<InverseInput> &batch) const;
    /** @brief Skips physical corrections for spectra already at training conditions.
     * Requires packaged labels/lengths. Packaged transforms still run, including
     * normalization embedded in ONNX; callers must not apply them again.
     */
    std::vector<InverseResult>
    predictPrepared(const std::vector<std::vector<simulator::Spectrum>> &batch) const;
    /** @brief Returns the validated package metadata used by this model. */
    const model::ModelMetadata &metadata() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ibeamlab::inference
