#pragma once
#include <ibeamlab/core/parameter.h>
#include <ibeamlab/core/spectrum.h>
#include <map>
#include <utility>
#include <variant>

namespace ibeamlab::model {
/** @brief Mapping represented by the graph: parameters to spectra, or spectra to EDP. */
enum class Direction { Forward, Inverse };
/** @brief Physical pileup state of reference-condition spectra. */
enum class SpectrumState { Bare, PiledUp };
/** @brief Reference spectrum shape, physical units and pileup state.
 * The matching reference measurement supplies the energy calibration, keV units,
 * particlesSr and real/live times. Numerical transforms never change this shape.
 */
struct SpectrumSpec {
    std::string measurement;
    std::size_t channels = 0;
    SpectrumState state = SpectrumState::Bare;
    std::string unit = "counts";
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
// Logical training operations. TransformPipeline declares direction and execution location.
using TensorTransform =
    std::variant<Log1p, MinMaxScale, StandardScale, MultiplyScale, CustomTransform>;
using TensorPipeline = std::vector<TensorTransform>;
/** @brief Apply stored training operations, or undo them in reverse order. */
enum class TransformDirection { Forward, Inverse };
/** @brief Runtime operations execute outside ONNX; Graph operations are already embedded. */
enum class TransformLocation { Runtime, Graph };
/** @brief Fitted numerical training transforms and their explicit execution boundary.
 * Inverse reverses step order and inverts each operation. Coefficients are never
 * refitted. Graph placement is a declaration by the exporter; runtime skips it.
 */
struct TransformPipeline {
    TensorPipeline steps;
    TransformDirection direction = TransformDirection::Forward;
    TransformLocation location = TransformLocation::Runtime;
};
/** @brief Independent numerical transforms for one labelled spectrum. */
struct SpectrumTransform {
    std::string measurement;
    TransformPipeline transform;
};
using SpectrumTransforms = std::vector<SpectrumTransform>;
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
using SpectrumOperation = std::variant<ExposureScale, Rebin, Pileup, RemovePileup>;
/** @brief Ordered operations for one measurement label; omitted labels are unchanged. */
struct SpectrumProcessing {
    std::string measurement;
    std::vector<SpectrumOperation> steps;
};
using SpectrumPipeline = std::vector<SpectrumProcessing>;
/** @brief Decode physical EDP outputs with a fixed maximum number of layers.
 * Graph uncertainty packs means, presence probabilities and posterior standard
 * deviations. Numerical output transforms affect means/std, never probabilities.
 */
struct EdpSchema {
    std::vector<std::string> elements;
    std::size_t layers = 0;
    bool uncertainty = false;
    std::string unit = "1e15 atoms/cm2";
};
/** @brief Fixed-width physical parameters for architectures that require a fixed sample.
 * Columns follow parameters. Unvaried sample properties come from reference_sample.
 */
struct ParameterInputSchema {
    core::Sample reference_sample;
    std::vector<core::ParameterSpec> parameters;
};
/** @brief Variable-length surface-to-depth sequence of elemental areal densities.
 * Graph inputs are float32 [batch,layers,elements]. Physical densities use
 * 1e15 atoms/cm2, with exactly one feature per ordered element and no thickness
 * column. No layer count is stored. Missing supported elements are zero; unknown
 * elements and explicit isotopes/roughness/porosity are unsupported.
 * input_transform runs independently per nonempty layer; zero padding stays zero.
 * Coefficients broadcast from one value or have one value per element.
 */
struct LayerInputSchema {
    std::vector<std::string> elements;
};
using ForwardInputSchema = std::variant<ParameterInputSchema, LayerInputSchema>;
/** @brief Architecture-independent forward input contract and labelled spectra.
 * reference_setup defines fixed experimental conditions, independent of sample
 * layer count. input_preparation defines graph rank and physical packing;
 * input_transform and output_transform describe fitted training operations.
 * output_preparation contains only measurement corrections, in execution order.
 */
struct IBEAMLAB_API ForwardModelMetadata {
    core::ExperimentalSetup reference_setup;
    ForwardInputSchema input_preparation;
    SpectraSchema output_spectra;
    TransformPipeline input_transform;
    SpectrumTransforms output_transform;
    SpectrumPipeline output_preparation;
};
/** @brief Contract for a multi-spectrum-to-elemental-depth-profile model.
 * input_preparation maps measured spectra to the reference setup, followed by
 * schema packing and per-detector input_transform. Graph outputs are flat [layer,element]
 * means, followed by probabilities and standard deviations when uncertainty is
 * enabled. output_transform restores physical densities; output_preparation
 * decodes their layout/units. Nonlinear uncertainty transforms are unsupported.
 */
struct IBEAMLAB_API InverseModelMetadata {
    core::ExperimentalSetup reference_setup;
    SpectraSchema input_spectra;
    SpectrumPipeline input_preparation;
    SpectrumTransforms input_transform;
    TransformPipeline output_transform;
    EdpSchema output_preparation;
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
    /** @brief Last-axis feature count: parameters, elements or packed spectrum channels. */
    std::size_t input_width() const;
    /** @brief True for a variable-length [batch,layers,elements] forward graph. */
    bool layer_input() const noexcept;
    /** @brief Graph input rank, 3 for layer sequences and 2 otherwise. */
    std::size_t input_rank() const noexcept { return layer_input() ? 3 : 2; }
    /** @brief Feature count of one graph output row, before output transforms and preparation. */
    std::size_t output_width() const;
    void validate() const;
};
} // namespace ibeamlab::model
