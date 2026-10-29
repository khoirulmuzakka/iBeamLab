#pragma once
#include <filesystem>
#include <ibeamlab/model/metadata.h>

namespace ibeamlab::model {
/** @brief Scientific metadata and an embedded standalone ONNX graph.
 * Format 1 is bounded, checksummed and little-endian, and stores only the active
 * directional schema. ONNX external tensor files are unsupported.
 */
class IBEAMLAB_API Package {
  public:
    ModelMetadata metadata;
    std::vector<std::byte> graph;
    /** @brief Check metadata and graph storage; graph execution is checked at inference. */
    void validate() const;
    /** @brief Serialize format 1; refuse to overwrite an existing file. */
    void write(const std::filesystem::path &) const;
    /** @brief Read a package and validate its envelope, checksum and metadata. */
    static Package open(const std::filesystem::path &);
    /** @brief Read standalone ONNX bytes and associate the supplied scientific contract. */
    static Package from_onnx(const std::filesystem::path &, ModelMetadata);
};
} // namespace ibeamlab::model
