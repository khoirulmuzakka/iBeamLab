#pragma once
#include <ibeamlab/fitting/optimizer.h>
#include <memory>
namespace ibeamlab::analysis {
/** Device and internal thread counts are supplied by the predictor factory.
 * Each task has a fresh objective; predictor/session resources persist across tasks. */
struct Task {
    std::string id;
    fitting::Problem problem;
    fitting::Options options;
    std::vector<fitting::Options> stages;
    std::vector<std::string> initial_from;
};
struct Outcome {
    std::string id;
    fitting::Result result;
    std::string error;
};
using PredictorFactory = std::function<std::shared_ptr<fitting::Predictor>(std::size_t worker)>;
using ProgressCallback =
    std::function<bool(const std::string &, std::size_t, const fitting::Progress &)>;
/** Independent fits with ordered outcomes, serialized progress callbacks, and
 * cooperative cancellation. Factories must return distinct worker predictors.
 * Completed dependency results seed later tasks when coordinate names match.
 * run() is serialized by caller; request_stop() is safe from another thread. */
class IBEAMLAB_API Executor {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    Executor(std::size_t workers, PredictorFactory);
    ~Executor();
    std::vector<Outcome> run(std::span<const Task>, ProgressCallback = {});
    void request_stop() noexcept;
};
} // namespace ibeamlab::analysis
