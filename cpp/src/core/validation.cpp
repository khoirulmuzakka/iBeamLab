#include <ibeamlab/core/validation.h>
#include <stdexcept>
namespace ibeamlab::core {
std::vector<ValidationIssue> validate(const Experiment &experiment) {
    std::vector<ValidationIssue> issues;
    auto check = [&](const auto &value, const char *path) {
        try {
            value.validate();
        } catch (const std::invalid_argument &error) {
            const std::string message = error.what();
            const auto separator = message.find(": ");
            issues.push_back(
                {separator == std::string::npos ? path : message.substr(0, separator),
                 separator == std::string::npos ? message : message.substr(separator + 2)});
        }
    };
    check(experiment.sample, "sample");
    check(experiment.setup, "setup");
    return issues;
}
} // namespace ibeamlab::core
