#pragma once
#include <ibeamlab/model/package.h>
#include <ibeamlab/inference/options.h>
#include <ibeamlab/simulation/simulator.h>
#include <memory>
namespace ibeamlab::inference {
/** @brief Measurement applies physical corrections; Reference skips them. */
enum class PredictionDomain { Measurement, Reference };
/** @brief Pack the declared physical input schema, run a graph and process spectra.
 * Instances reuse scratch buffers and are not concurrently callable: use one per
 * worker. Graph tensors use float32; physical results use double precision.
 */
class IBEAMLAB_API ForwardModel {
  public:
    explicit ForwardModel(model::Package, InferenceOptions = {});
    ~ForwardModel();
    /** @brief Predict into caller-owned output, reusing storage where possible.
     * Parameter/element bindings and permitted changes are resolved at construction.
     * Layer schemas accept any positive layer count and grow capacity only as needed.
     * Without spectrum processing, graph values convert directly into output.
     */
    void predict(const core::Experiment &, simulation::Result &output, PredictionDomain = PredictionDomain::Measurement);
    simulation::Result predict(const core::Experiment &, PredictionDomain = PredictionDomain::Measurement);
    simulation::Result predict_reference(const core::Experiment &x) { return predict(x, PredictionDomain::Reference); }
    void predict_reference(const core::Experiment &x, simulation::Result &out) { predict(x, out, PredictionDomain::Reference); }
    /** @brief Predict from host-resident float32 AD in contiguous [layers,elements] order.
     * Requires a LayerInputSchema. No sample object or parameter list is built.
     * setup defaults to the reference setup. Caller memory is never modified;
     * preprocessing uses retained scratch storage, bypassed for an empty pipeline.
     */
    void predict_areal_density(std::span<const float> densities, std::size_t layers,
                              simulation::Result &output,
                              const core::ExperimentalSetup *setup = nullptr, PredictionDomain = PredictionDomain::Measurement);
    simulation::Result predict_areal_density(std::span<const float> densities, std::size_t layers,
                                            const core::ExperimentalSetup *setup = nullptr, PredictionDomain = PredictionDomain::Measurement);
    /** @brief Direct AD to physical reference spectra, bypassing detector corrections. */
    simulation::Result predict_reference_areal_density(std::span<const float> ad, std::size_t layers) {
        return predict_areal_density(ad, layers, nullptr, PredictionDomain::Reference);
    }
    void predict_reference_areal_density(std::span<const float> ad, std::size_t layers, simulation::Result &out) {
        predict_areal_density(ad, layers, out, nullptr, PredictionDomain::Reference);
    }
    /** @brief Run the population in one ONNX call; zero-pad differing layer counts.
     * Results preserve input order and each experiment's processing setup.
     */
    std::vector<simulation::Result> predict_batch(std::span<const core::Experiment>, PredictionDomain = PredictionDomain::Measurement);
    /** @brief Batched prediction into reusable caller-owned result storage. */
    void predict_batch(std::span<const core::Experiment>, std::span<simulation::Result>, PredictionDomain = PredictionDomain::Measurement);
    std::vector<simulation::Result> predict_reference_batch(std::span<const core::Experiment> x) { return predict_batch(x, PredictionDomain::Reference); }
    void predict_reference_batch(std::span<const core::Experiment> x, std::span<simulation::Result> out) { predict_batch(x, out, PredictionDomain::Reference); }
    /** @brief Processed channel counts, in packaged spectrum order. */
    std::span<const std::size_t> spectrum_channels(PredictionDomain = PredictionDomain::Measurement) const;
    /** @brief One graph call for physical AD [batch,layers,elements].
     * Output is caller-owned float64 [batch,sum(spectrum_channels())], after
     * detector-specific processing. setup applies to the whole fitting population.
     * Retains graph/preprocessing workspaces; borrows AD if no transform is needed.
     * Empty batches do not execute the graph. Input and output must not overlap.
     */
    void predict_areal_density_batch(std::span<const float>, std::size_t batch,
                                    std::size_t layers, std::span<double> output,
                                    const core::ExperimentalSetup *setup = nullptr, PredictionDomain = PredictionDomain::Measurement);
    /** @brief One ONNX population call; caller-owned output contains reference spectra. */
    void predict_reference_areal_density_batch(std::span<const float> ad, std::size_t batch,
                                              std::size_t layers, std::span<double> out) {
        predict_areal_density_batch(ad, batch, layers, out, nullptr, PredictionDomain::Reference);
    }
    const model::ModelMetadata &metadata() const;
    const InferenceOptions &options() const;

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
    explicit InverseModel(model::Package, InferenceOptions = {});
    ~InverseModel();
    /** @brief Run spectrum preparation, tensor preprocessing, graph and EDP processing. */
    void predict(const InverseInput &, Prediction &output);
    /** @brief Use spectra already on the reference grid and exposure.
     * Skips input_preparation only; tensor preprocessing and EDP processing still
     * run. Labels and channel counts must match the input schema.
     */
    void predict_reference(std::span<const core::Spectrum>, Prediction &output);
    Prediction predict_reference(std::span<const core::Spectrum> spectra) {
        Prediction out;
        predict_reference(spectra, out);
        return out;
    }
    Prediction predict(const InverseInput &);
    /** @brief Prepare each input, then execute one ONNX call for the batch. */
    std::vector<Prediction> predict_batch(std::span<const InverseInput>, PredictionDomain = PredictionDomain::Measurement);
    /** @brief Batched inference retaining caller-owned prediction storage. */
    void predict_batch(std::span<const InverseInput>, std::span<Prediction>, PredictionDomain = PredictionDomain::Measurement);
    std::vector<Prediction> predict_reference_batch(std::span<const InverseInput> x) { return predict_batch(x, PredictionDomain::Reference); }
    void predict_reference_batch(std::span<const InverseInput> x, std::span<Prediction> out) { predict_batch(x, out, PredictionDomain::Reference); }
    const model::ModelMetadata &metadata() const;
    const InferenceOptions &options() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ibeamlab::inference
