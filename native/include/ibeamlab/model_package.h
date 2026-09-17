#pragma once

/**
 * @file model_package.h
 * @brief Reads and writes portable ONNX model packages.
 *
 * C++ owns the stable ZIP/TOML/ONNX interchange format, validation, checksums,
 * and safe I/O. Python training code decides the model and metadata contents.
 */
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <ibeamlab/generation.h>
#include <ibeamlab/sample.h>
#include <string>
#include <vector>

namespace ibeamlab::model {
/** @brief Serializable description of one preprocessing transform or pipeline. */
struct TransformSpec { std::string type{"identity"}; std::size_t inputDimension{}; float factor{1}; float offset{1}; float low{0}; float high{1}; std::vector<float> minimum,scale,mean,deviation; std::vector<TransformSpec> transforms; };
enum class ModelType { Inverse, Forward };
struct SpectrumSpec {
    std::string label;
    std::size_t length{};
};
struct InverseModelMetadata {
    sample::SampleModel sampleTemplate;
    sample::ExperimentalSetup setupTemplate;
    std::vector<SpectrumSpec> inputSpectra;
    std::vector<generation::ParameterSpec> outputParameters;
};
struct ForwardModelMetadata {
    sample::SampleModel sampleTemplate;
    sample::ExperimentalSetup setupTemplate;
    std::vector<generation::ParameterSpec> inputParameters;
    std::vector<SpectrumSpec> outputSpectra;
};
/** @brief Versioned metadata for exactly one inverse or forward model. */
struct ModelMetadata {
    std::uint32_t formatVersion{2}; std::string createdUtc,className;
    ModelType modelType{ModelType::Inverse};
    std::string inputName{"inputs"},outputName{"outputs"}; std::size_t inputDimension{},outputDimension{}; int opsetVersion{};
    std::string modelChecksum;
    std::uint64_t modelSize{};
    TransformSpec inputTransform,outputTransform;
    InverseModelMetadata inverse;
    ForwardModelMetadata forward;
};

/**
 * @brief Validated in-memory ONNX model and its package metadata.
 *
 * Packages may be opened from a development directory or ZIP and written to
 * either form. The writer calculates model size and CRC32 itself.
 */
class ModelPackage {
public:
    /** @brief Opens and validates a package directory or ZIP archive. */
    static ModelPackage open(const std::filesystem::path& path);
    /** @brief Creates a validated package from an ONNX file and metadata. */
    static ModelPackage fromOnnx(const std::filesystem::path& onnxFile,
                                 ModelMetadata metadata);
    /** @brief Writes a new directory package or ZIP without overwriting. */
    void write(const std::filesystem::path& path) const;
    /** @brief Returns immutable package metadata. */
    const ModelMetadata& metadata() const noexcept{return metadata_;}
    /** @brief Returns the validated ONNX payload. */
    const std::vector<std::byte>& modelBytes() const noexcept{return modelBytes_;}
private: ModelMetadata metadata_;std::vector<std::byte> modelBytes_;
};
}
