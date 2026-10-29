#pragma once
#include <ibeamlab/model/package.h>
#include <memory>
namespace ibeamlab::inference::detail {
IBEAMLAB_API void run_graph(const model::Package &, std::span<float>, std::span<float>,
                                 std::size_t batch, int threads);
class Session {
  public:
    Session(const model::Package &, int threads);
    ~Session();
    void run(std::span<float> input, std::span<float> output, std::size_t batch = 1);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ibeamlab::inference::detail
