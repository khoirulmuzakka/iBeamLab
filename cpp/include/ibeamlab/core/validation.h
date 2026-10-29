#pragma once
#include <ibeamlab/core/experiment.h>
#include <string>
#include <vector>
namespace ibeamlab::core {
/** @brief Validation diagnostic with a field path identifying the invalid input. */
struct ValidationIssue {
    std::string path;
    std::string message;
};
// Collect sample and setup errors independently; does not mutate the input.
IBEAMLAB_API std::vector<ValidationIssue> validate(const Experiment &experiment);
} // namespace ibeamlab::core
