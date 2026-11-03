#include <algorithm>
#include <ibeamlab/fitting/optimizer.h>
#include <limits>
#include <stdexcept>
#include <arrde.h>
#include <de.h>
namespace ibeamlab::fitting {
bool arrde_available() noexcept {
    return true;
}
Result fit(Objective &objective, Options options, ProgressCallback callback) {
    if (options.algorithm != "ARRDE" && options.algorithm != "DE")
        throw std::invalid_argument("unsupported optimizer: " + options.algorithm);
    for (const auto &row : options.initial_guesses) {
        if (row.size() != objective.dimension())
            throw std::invalid_argument("initial guess dimension differs");
        for (std::size_t j = 0; j < row.size(); ++j)
            if (!objective.bounds()[j].contains(row[j]))
                throw std::invalid_argument("initial guess outside bounds");
    }
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
    std::vector<std::pair<double, double>> bounds;
    for (const auto &b : objective.bounds())
        bounds.emplace_back(b.minimum, b.maximum);
    std::vector<std::vector<double>> initial{
        {objective.initial_coordinates().begin(), objective.initial_coordinates().end()}};
    for (const auto &row : options.initial_guesses)
        initial.push_back(row);
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
    for (const auto &[name, value] : options.settings)
        settings[name] = value;
    if (options.population_size)
        settings["population_size"] = int(options.population_size);
    auto callback_function = [&](minion::MinionResult *) { return report(); };
    minion::MinionResult result;
    if (options.algorithm == "DE") {
        minion::Differential_Evolution solver(
            function, bounds, initial, nullptr, callback_function,
            options.max_evaluations - objective.progress().evaluations, options.seed, settings);
        result = solver.optimize();
    } else {
        minion::ARRDE solver(function, bounds, initial, nullptr, callback_function,
                             options.max_evaluations - objective.progress().evaluations,
                             options.seed, settings);
        result = solver.optimize();
    }
    if (options.record_history &&
        (history.empty() || history.back().evaluations != objective.progress().evaluations))
        history.push_back(objective.progress());
    return finish(minion::terminationStatusToString(result.status), result.message);
}
Result fit_pipeline(Objective &objective, std::span<const Options> stages,
                    ProgressCallback callback) {
    if (stages.empty())
        throw std::invalid_argument("optimizer pipeline is empty");
    Result result;
    std::vector<Progress> history;
    for (std::size_t i = 0; i < stages.size(); ++i) {
        auto settings = stages[i];
        if (!settings.max_evaluations)
            throw std::invalid_argument("stage budget must be positive");
        if (i) {
            if (settings.max_evaluations >
                std::numeric_limits<std::size_t>::max() - objective.progress().evaluations)
                throw std::invalid_argument("pipeline evaluation budget overflow");
            settings.max_evaluations += objective.progress().evaluations;
            auto best = objective.best_coordinates();
            settings.initial_guesses.insert(settings.initial_guesses.begin(),
                                            core::Counts(best.begin(), best.end()));
        }
        result = fit(objective, settings, callback);
        history.insert(history.end(), result.history.begin(), result.history.end());
        if (objective.stopped())
            break;
    }
    result.history = std::move(history);
    return result;
}
} // namespace ibeamlab::fitting
