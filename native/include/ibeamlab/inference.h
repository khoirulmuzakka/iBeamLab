#pragma once
/** @file inference.h @brief Shared execution options for forward and inverse ONNX models. */
#include <optional>
#include <string>
namespace ibeamlab::inference {
struct InferenceOptions {
    std::optional<bool> applyPileupOnInference; // nullopt uses package policy.
    int correctionThreads{1}; // Forward batch corrections; independent of ONNX threads.
    int intraOpThreads{1};
    int interOpThreads{1};
    std::string executionProvider{"cpu"};
    bool enableGraphOptimizations{false};
};
} // namespace ibeamlab::inference
