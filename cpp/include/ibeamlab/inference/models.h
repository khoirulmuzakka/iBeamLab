#pragma once
#include <ibeamlab/model/package.h>
#include <ibeamlab/simulation/simulator.h>
#include <memory>
namespace ibeamlab::inference {
/** @brief Pack physical parameters, run a forward graph and process labelled spectra.
 * Instances reuse scratch buffers and are not concurrently callable: use one per
 * worker. Graph tensors use float32; physical results use double precision.
 */
class IBEAMLAB_API ForwardModel {
  public:
    explicit ForwardModel(model::Package, int threads = 1);
    ~ForwardModel();
    /** @brief Predict into caller-owned output, reusing storage where possible.
     * Parameter bindings and permitted changes are resolved at construction.
     * Without spectrum processing, graph values convert directly into output.
     */
    void predict(const core::Experiment &, simulation::Result &output);
    simulation::Result predict(const core::Experiment &);
    /** @brief Predict sequentially in input order using this instance's reusable buffers. */
    std::vector<simulation::Result> predict_batch(std::span<const core::Experiment>);
    const model::ModelMetadata &metadata() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
/** @brief Measured spectra and the setup under which they were acquired.
 * Labels match the package schema; caller spectrum order is arbitrary.
 */
struct InverseInput {
    core::ExperimentalSetup setup;
    std::vector<core::Spectrum> spectra;
    /// Skip the first requested-acquisition RemovePileup step for each spectrum.
    bool pileup_already_removed = false;
};
/** @brief Elemental areal densities and optional model-predicted uncertainty.
 * Arrays are row-major [layer,element], surface to depth. Without graph uncertainty,
 * standard deviations are zero and presence is 1 for positive density, 0 otherwise.
 */
struct Prediction {
    std::vector<std::string> elements;
    std::size_t layers = 0;
    core::Counts values, presence_probability, posterior_std;
    bool uncertainty_predicted = false;
    // All matrices are flat [layer,element], surface-to-depth.
    std::string unit = "1e15 atoms/cm2";
};
/** @brief Prepare labelled spectra and infer an elemental depth profile.
 * Instances retain reusable buffers and must not be called concurrently. Use one
 * instance per worker. Custom preprocessing kernels must be registered beforehand.
 */
class IBEAMLAB_API InverseModel {
  public:
    explicit InverseModel(model::Package, int threads = 1);
    ~InverseModel();
    /** @brief Run spectrum preparation, tensor preprocessing, graph and EDP processing. */
    void predict(const InverseInput &, Prediction &output);
    /** @brief Use spectra already on the reference grid and exposure.
     * Skips input_preparation only; tensor preprocessing and EDP processing still
     * run. Labels and channel counts must match the input schema.
     */
    void predict_prepared(std::span<const core::Spectrum>, Prediction &output);
    Prediction predict(const InverseInput &);
    /** @brief Predict sequentially in input order using this instance's reusable buffers. */
    std::vector<Prediction> predict_batch(std::span<const InverseInput>);
    const model::ModelMetadata &metadata() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ibeamlab::inference
