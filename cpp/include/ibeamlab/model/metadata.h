#pragma once
#include <ibeamlab/core/parameter.h>
#include <ibeamlab/core/spectrum.h>
#include <map>
#include <utility>
#include <variant>

namespace ibeamlab::model {
/** @brief Mapping represented by the graph: parameters to spectra, or spectra to EDP. */
enum class Direction { Forward, Inverse };
/** @brief One labelled slice of a packed spectrum tensor, before spectrum processing. */
struct SpectrumSpec {
    std::string measurement;
    std::size_t channels = 0;
};
/** @brief Ordered layout of spectra in one packed graph tensor.
 * Spectra occupy consecutive slices of potentially different lengths. Labels
 * identify measurements independently of the order supplied by the caller.
 */
struct IBEAMLAB_API SpectraSchema {
    std::vector<SpectrumSpec> spectra;
    /** @brief Total channel count across all slices. */
    std::size_t width() const;
    /** @brief First feature index for a label; throws if the label is absent. */
    std::size_t offset(std::string_view measurement) const;
};
/** @brief Elementwise natural logarithm log(1 + x); requires nonnegative inputs. */
struct Log1p {};
/** @brief Fitted mapping low + (x-minimum)*(high-low)/(maximum-minimum).
 * Coefficients broadcast from one value or match the packed input width.
 * When clip is true, mapped values are clamped to [low, high].
 */
struct MinMaxScale {
    core::Counts minimum, maximum;
    double low = 0, high = 1;
    bool clip = false;
};
/** @brief Fitted standardization (x-mean)/deviation; deviations must be positive.
 * Each coefficient array broadcasts from one value or matches the input width.
 */
struct StandardScale {
    core::Counts mean, deviation;
};
/** @brief Elementwise multiplication by one broadcast factor or per-feature factors. */
struct MultiplyScale {
    core::Counts factors{1};
};
using CustomParameters = std::map<std::string, core::Counts>;
/** @brief Portable descriptor of an application-supplied, fixed-width transform.
 * The consuming C++ application registers a kernel under identifier and version.
 * Only named numeric parameters are packaged; executable code is not embedded.
 */
struct CustomTransform {
    std::string identifier;
    std::uint32_t version = 1;
    CustomParameters parameters;
};
// Scalar coefficients broadcast; otherwise one value per ordered tensor feature.
// Operations run in place, in order, outside ONNX. Fitted values are never refit.
using TensorTransform =
    std::variant<Log1p, MinMaxScale, StandardScale, MultiplyScale, CustomTransform>;
using TensorPipeline = std::vector<TensorTransform>;
/** @brief Multiply every channel of one spectrum by a nonnegative factor. */
struct MultiplyCounts {
    double factor = 1;
};
/** @brief Scale by the ratio of particlesSr exposures, both of which must be positive.
 * Forward maps reference to requested exposure; inverse maps measured to reference.
 */
struct ExposureScale {};
/** @brief Redistribute integrated counts onto the destination detector calibration.
 * Destination is the requested calibration for forward inference, reference for
 * inverse inference. Zero channels selects the measurement's schema channel count.
 * Full coverage requires the source energy range to cover the destination grid.
 */
struct Rebin {
    std::size_t channels = 0;
    bool require_full_coverage = true;
};
/** @brief Select acquisition times from the inference request or package reference. */
enum class AcquisitionSource { Requested, Reference };
/** @brief Apply the pair-sum pileup model, expanding N channels to 2*N-1.
 * tau_seconds is the resolving time; acquisition selects real/live times.
 */
struct Pileup {
    double tau_seconds = 0;
    AcquisitionSource acquisition = AcquisitionSource::Requested;
};
/** @brief Invert pair-sum pileup from complete 2*N-1 support to N channels.
 * Reconstruction must satisfy relative_tolerance; truncated or incompatible
 * spectra are rejected rather than silently approximated.
 */
struct RemovePileup {
    double tau_seconds = 0;
    double relative_tolerance = 1e-7;
    AcquisitionSource acquisition = AcquisitionSource::Requested;
};
using SpectrumOperation = std::variant<MultiplyCounts, ExposureScale, Rebin, Pileup, RemovePileup>;
/** @brief Ordered operations for one measurement label; omitted labels are unchanged. */
struct SpectrumProcessing {
    std::string measurement;
    std::vector<SpectrumOperation> steps;
};
using SpectrumPipeline = std::vector<SpectrumProcessing>;
/** @brief Multiply EDP means and standard deviations by a positive factor.
 * Presence probabilities remain dimensionless and unchanged.
 */
struct ScaleEdp {
    double factor = 1;
};
using EdpPipeline = std::vector<ScaleEdp>;
/** @brief Contract for a parameter-to-spectrum surrogate.
 * Physical input columns follow input_parameters, then preprocessing runs before
 * the graph. Graph output slices follow output_spectra; postprocessing operates
 * on each labelled spectrum afterwards. Unvaried physics comes from reference.
 */
struct IBEAMLAB_API ForwardModelMetadata {
    core::Experiment reference;
    std::vector<core::ParameterSpec> input_parameters;
    SpectraSchema output_spectra;
    TensorPipeline preprocessing;
    SpectrumPipeline postprocessing;
};
/** @brief Contract for a multi-spectrum-to-elemental-depth-profile model.
 * input_preparation maps measured spectra to the reference setup, followed by
 * schema packing and tensor preprocessing. Graph outputs are flat [layer,element]
 * means, followed by probabilities and standard deviations when uncertainty is
 * enabled. postprocessing converts density outputs to physical areal densities.
 */
struct IBEAMLAB_API InverseModelMetadata {
    core::ExperimentalSetup reference_setup;
    SpectraSchema input_spectra;
    std::vector<std::string> elements;
    std::size_t layers = 0;
    bool uncertainty = false;
    SpectrumPipeline input_preparation;
    TensorPipeline preprocessing;
    EdpPipeline postprocessing;
};
/** @brief Direction-specific scientific contract and shared graph tensor names.
 * Direction is derived from the active variant. validate() checks physical
 * settings, dimensions, labels and processing descriptors, but does not execute
 * the graph or resolve application-supplied custom transform kernels.
 */
struct IBEAMLAB_API ModelMetadata {
    std::variant<ForwardModelMetadata, InverseModelMetadata> model;
    std::string input_name = "inputs", output_name = "outputs", provenance;
    ModelMetadata() = default;
    explicit ModelMetadata(ForwardModelMetadata value) : model(std::move(value)) {}
    explicit ModelMetadata(InverseModelMetadata value) : model(std::move(value)) {}
    Direction direction() const noexcept;
    const core::ExperimentalSetup &reference_setup() const;
    const std::vector<SpectrumSpec> &spectra() const;
    const TensorPipeline &preprocessing() const;
    /** @brief Feature count of one graph input row, excluding the batch dimension. */
    std::size_t input_width() const;
    /** @brief Feature count of one graph output row, before postprocessing. */
    std::size_t output_width() const;
    void validate() const;
};
} // namespace ibeamlab::model
