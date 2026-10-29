#pragma once
#include <ibeamlab/model/package.h>
#include <ibeamlab/inference/options.h>
#include <memory>
namespace ibeamlab::inference::detail {
IBEAMLAB_API void run_graph(const model::Package &, std::span<const float>, std::span<float>,
                           std::size_t batch, InferenceOptions options = {}, std::size_t layers = 1);
class Session {
  public:
    Session(const model::Package &, InferenceOptions);
    ~Session();
    void run(std::span<const float> input, std::span<float> output, std::size_t batch = 1,
             std::size_t layers = 1);
    const InferenceOptions &options() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    InferenceOptions options_;
};
} // namespace ibeamlab::inference::detail
