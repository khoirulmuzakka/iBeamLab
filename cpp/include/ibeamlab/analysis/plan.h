#pragma once
#include <ibeamlab/fitting/predictor.h>
namespace ibeamlab::analysis {
/** Route one or more measurement labels to a worker-owned batch predictor.
 * Route labels are both predictor input setup and selected output labels. */
struct Route {
    std::vector<std::string> measurements;
    std::shared_ptr<fitting::Predictor> predictor;
};
struct IBEAMLAB_API Plan {
    std::vector<Route> routes;
    void validate() const;
};
} // namespace ibeamlab::analysis
