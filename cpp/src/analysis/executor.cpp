#include <atomic>
#include <condition_variable>
#include <deque>
#include <ibeamlab/analysis/executor.h>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
namespace ibeamlab::analysis {
struct Executor::Impl {
    std::vector<std::shared_ptr<fitting::Predictor>> predictors;
    std::vector<fitting::Objective *> active;
    std::atomic_bool stop = false;
    std::mutex mutex, callbacks;
    std::condition_variable ready;
};
Executor::Executor(std::size_t workers, PredictorFactory factory)
    : impl_(std::make_unique<Impl>()) {
    if (!workers || workers > 256 || !factory)
        throw std::invalid_argument("invalid analysis worker factory/count");
    std::set<const fitting::Predictor *> used;
    for (std::size_t i = 0; i < workers; ++i) {
        auto p = factory(i);
        if (!p || !used.insert(p.get()).second)
            throw std::invalid_argument("each analysis worker requires a distinct predictor");
        impl_->predictors.push_back(std::move(p));
    }
    impl_->active.resize(workers, nullptr);
}
Executor::~Executor() = default;
void Executor::request_stop() noexcept {
    impl_->stop = true;
    std::lock_guard lock(impl_->mutex);
    for (auto &p : impl_->predictors)
        p->request_stop();
    for (auto *o : impl_->active)
        if (o)
            o->request_stop();
    impl_->ready.notify_all();
}
std::vector<Outcome> Executor::run(std::span<const Task> tasks, ProgressCallback callback) {
    auto &a = *impl_;
    if (a.stop)
        throw std::logic_error("cancelled executor must be recreated");
    std::map<std::string, std::size_t> lookup;
    for (std::size_t i = 0; i < tasks.size(); ++i)
        if (tasks[i].id.empty() || !lookup.emplace(tasks[i].id, i).second)
            throw std::invalid_argument("analysis task IDs must be unique");
    if (tasks.empty())
        return {};
    std::vector<std::vector<std::size_t>> dependents(tasks.size());
    std::vector<std::size_t> remaining(tasks.size());
    std::deque<std::size_t> ready;
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        std::set<std::string> seen;
        for (const auto &id : tasks[i].initial_from) {
            if (!lookup.contains(id) || lookup.at(id) >= i || !seen.insert(id).second)
                throw std::invalid_argument(
                    "initialization dependencies must be unique and precede task");
            ++remaining[i];
            dependents[lookup.at(id)].push_back(i);
        }
        if (!remaining[i])
            ready.push_back(i);
    }
    std::vector<Outcome> results(tasks.size());
    std::vector<bool> done(tasks.size());
    std::size_t completed = 0;
    std::vector<std::vector<std::string>> names(tasks.size());
    auto worker = [&](std::size_t w) {
        for (;;) {
            std::size_t index;
            {
                std::unique_lock lock(a.mutex);
                a.ready.wait(lock,
                             [&] { return a.stop || !ready.empty() || completed == tasks.size(); });
                if (a.stop || completed == tasks.size())
                    return;
                index = ready.front();
                ready.pop_front();
            }
            auto &out = results[index];
            const auto &task = tasks[index];
            out.id = task.id;
            try {
                {
                    std::lock_guard lock(a.mutex);
                    if (!a.stop)
                        a.predictors[w]->reset_stop();
                }
                fitting::Objective objective(task.problem, a.predictors[w]);
                auto options = task.options;
                names[index].assign(objective.coordinate_names().begin(),
                                    objective.coordinate_names().end());
                {
                    std::lock_guard lock(a.mutex);
                    for (const auto &id : task.initial_from) {
                        auto j = lookup.at(id);
                        if (results[j].result.feasible() && names[j] == names[index]) {
                            const auto &row = results[j].result.coordinates;
                            bool valid = row.size() == objective.dimension();
                            for (std::size_t k = 0; valid && k < row.size(); ++k)
                                valid = objective.bounds()[k].contains(row[k]);
                            if (valid)
                                options.initial_guesses.push_back(row);
                        }
                    }
                    a.active[w] = &objective;
                    if (a.stop)
                        objective.request_stop();
                }
                try {
                    auto report = [&](const auto &p) {
                        if (a.stop)
                            return true;
                        if (!callback)
                            return false;
                        std::lock_guard lock(a.callbacks);
                        return callback(task.id, w, p);
                    };
                    if (task.stages.empty())
                        out.result = fitting::fit(objective, options, report);
                    else {
                        auto stages = task.stages;
                        stages.front().initial_guesses.insert(stages.front().initial_guesses.end(),
                                                              options.initial_guesses.begin(),
                                                              options.initial_guesses.end());
                        out.result = fitting::fit_pipeline(objective, stages, report);
                    }
                } catch (...) {
                    std::lock_guard lock(a.mutex);
                    a.active[w] = nullptr;
                    throw;
                }
                {
                    std::lock_guard lock(a.mutex);
                    a.active[w] = nullptr;
                }
            } catch (const std::exception &e) {
                out.error = e.what();
            } catch (...) {
                out.error = "unknown analysis worker failure";
            }
            {
                std::lock_guard lock(a.mutex);
                done[index] = true;
                ++completed;
                for (auto next : dependents[index])
                    if (--remaining[next] == 0)
                        ready.push_back(next);
                a.ready.notify_all();
            }
        }
    };
    std::vector<std::jthread> threads;
    for (std::size_t w = 0; w < a.predictors.size(); ++w)
        threads.emplace_back(worker, w);
    for (auto &t : threads)
        t.join();
    for (std::size_t i = 0; i < tasks.size(); ++i)
        if (!done[i]) {
            results[i].id = tasks[i].id;
            results[i].result.status = "cancelled";
        }
    return results;
}
} // namespace ibeamlab::analysis
