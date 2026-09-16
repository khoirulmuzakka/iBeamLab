#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ibeamlab::model {
struct TransformSpec { std::string type{"identity"}; std::size_t inputDimension{}; float factor{1}; float offset{1}; float low{0}; float high{1}; std::vector<float> minimum,scale,mean,deviation; std::vector<TransformSpec> transforms; };
struct ModelMetadata {
    std::uint32_t formatVersion{}; std::string createdUtc,task,className;
    std::string inputName{"inputs"},outputName{"outputs"}; std::size_t inputDimension{},outputDimension{}; int opsetVersion{};
    std::vector<std::string> methodNames,inputFeatures,outputFeatures,outputUnits;
    std::vector<std::size_t> spectrumLengths;
    std::string modelChecksum;
    std::uint64_t modelSize{};
    TransformSpec inputTransform,outputTransform;
};
class ModelPackage {
public:
    static ModelPackage open(const std::filesystem::path& path);
    const ModelMetadata& metadata() const noexcept{return metadata_;}
    const std::vector<std::byte>& modelBytes() const noexcept{return modelBytes_;}
private: ModelMetadata metadata_;std::vector<std::byte> modelBytes_;
};
}
