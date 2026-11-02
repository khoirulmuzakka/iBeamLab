#include <ibeamlab/analysis/plan.h>
#include <set>
#include <stdexcept>
namespace ibeamlab::analysis {
void Plan::validate() const {
    if (routes.empty())
        throw std::invalid_argument("analysis needs at least one route");
    std::set<std::string> labels;
    std::set<const fitting::Predictor *> providers;
    for (const auto &r : routes) {
        if (!r.predictor || r.measurements.empty() || !providers.insert(r.predictor.get()).second)
            throw std::invalid_argument("routes require distinct predictors and nonempty labels");
        for (const auto &label : r.measurements)
            if (label.empty() || !labels.insert(label).second)
                throw std::invalid_argument("duplicate/empty routed measurement");
    }
}
} // namespace ibeamlab::analysis
