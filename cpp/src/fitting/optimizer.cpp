#include <algorithm>
#include <ibeamlab/fitting/optimizer.h>
#include <limits>
#include <stdexcept>
#ifdef IBEAMLAB_HAS_ARRDE
#include <arrde.h>
#endif
namespace ibeamlab::fitting {
bool arrde_available() noexcept {
#ifdef IBEAMLAB_HAS_ARRDE
    return true;
#else
    return false;
#endif
}
Result fit(Objective &objective, Options options, ProgressCallback callback) {
    if (!options.max_evaluations)
        throw std::invalid_argument("max_evaluations must be positive");
    if (options.population_size &&
        (options.population_size < 4 ||
         options.population_size > std::size_t(std::numeric_limits<int>::max())))
        throw std::invalid_argument("population_size must be zero or at least four and fit in int");
    std::vector<Progress> history;
    auto report = [&]() {
        auto p = objective.progress();
        if (options.record_history)
            history.push_back(p);
        if (callback && callback(p))
            objective.request_stop();
        return objective.stopped();
    };
    auto finish = [&](std::string status, std::string message) {
        auto r = objective.best_result();
        r.status = objective.stopped() ? "cancelled" : std::move(status);
        r.message = std::move(message);
        r.history = std::move(history);
        return r;
    };
    if (report())
        return finish("cancelled", "Stopped before optimization");
    if (!objective.dimension())
        return finish("fixed", "All parameters are fixed");
    if (objective.progress().evaluations >= options.max_evaluations)
        return finish("max_evaluations_reached", "Evaluation budget exhausted");
#ifdef IBEAMLAB_HAS_ARRDE
    std::vector<std::pair<double, double>> bounds;
    for (const auto &b : objective.bounds())
        bounds.emplace_back(b.minimum, b.maximum);
    std::vector<std::vector<double>> initial{
        {objective.initial_coordinates().begin(), objective.initial_coordinates().end()}};
    std::vector<double> flat;
    auto function = [&](const std::vector<std::vector<double>> &rows, void *) {
        flat.resize(rows.size() * objective.dimension());
        for (std::size_t j = 0; j < rows.size(); ++j)
            std::copy(rows[j].begin(), rows[j].end(), flat.begin() + j * objective.dimension());
        std::vector<double> costs(rows.size(), INFINITY);
        auto used = objective.progress().evaluations;
        auto remaining = used >= options.max_evaluations ? 0 : options.max_evaluations - used;
        auto count = std::min(remaining, rows.size());
        if (count)
            objective.evaluate_batch(
                std::span<const double>(flat.data(), count * objective.dimension()), count,
                std::span<double>(costs.data(), count));
        return costs;
    };
    std::map<std::string, minion::ConfigValue> settings;
    if (options.population_size)
        settings["population_size"] = int(options.population_size);
    minion::ARRDE solver(
        function, bounds, initial, nullptr, [&](minion::MinionResult *) { return report(); },
        options.max_evaluations - objective.progress().evaluations, options.seed, settings);
    auto result = solver.optimize();
    if (options.record_history &&
        (history.empty() || history.back().evaluations != objective.progress().evaluations))
        history.push_back(objective.progress());
    return finish(minion::terminationStatusToString(result.status), result.message);
#else
    throw std::runtime_error(
        "ARRDE is unavailable; configure IBEAMLAB_ENABLE_ARRDE=ON or use an external optimizer");
#endif
}
} // namespace ibeamlab::fitting
